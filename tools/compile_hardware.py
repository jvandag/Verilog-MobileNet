"""Compile the pinned graph into a fixed-profile RTL ROM and xsim fixtures.

Run from the submodule: python tools/compile_hardware.py [--image photo.jpg ...]
Configuration and immutable constants are compiled into the bitstream.
"""
import argparse
import hashlib
import json
import math
import zlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'models/mobilenet025-cifar4/integer'


def multiplier(scale):
    fraction, shift = math.frexp(float(scale))
    q = int(math.floor(fraction * (1 << 31) + .5))
    if q == 1 << 31:
        q //= 2
        shift += 1
    assert -31 <= shift <= 0, (scale, shift)
    return q, shift


def requantize(x, q, shift):
    # TensorFlow 2.20.0 wheel's single-rounding integer kernel.
    total_shift = 31-shift
    return (int(x)*int(q)+(1 << (total_shift-1))) >> total_shift


def mem(path, values, bits=8):
    path.write_text(''.join(f'{int(v) & ((1 << bits)-1):0{bits//4}x}\n'
                            for v in np.asarray(values).reshape(-1)), encoding='ascii')


def compile_profile():
    m = json.loads((BUNDLE / 'manifest.json').read_text())
    raw = (BUNDLE / 'constants.bin').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == m['constants_sha256']
    assert hashlib.sha256((BUNDLE / 'model.tflite').read_bytes()).hexdigest() == m['model_sha256']
    assert m['input_shape'] == [1, 64, 64, 3] and len(m['operators']) == 30
    assert m['constant_bytes'] == len(raw) == 219592
    assert m['classes'] == ['bird','cat','dog','horse']
    assert m['macs_per_image'] == 3329536
    assert len(m['input_pixel_lut']) == 256 and all(0 <= v <= 255 for v in m['input_pixel_lut'])
    t = {v['index']: v for v in m['tensors']}
    dest = ROOT / 'rtl/generated'
    dest.mkdir(parents=True, exist_ok=True)
    fields = 'kind,src_bank,dst_bank,ih,iw,ic,oh,ow,oc,kh,kw,stride,pady,padx,iz,oz,wbase,bbase,qbase,clo,chi,lgiw,lgic,lgow,lgoc'
    cases = []
    qparams = []
    descriptors = []
    for n, op in enumerate(m['operators']):
        expected_kind = 'QUANTIZE' if n==0 else 'MEAN' if n==28 else 'FULLY_CONNECTED' if n==29 else 'DEPTHWISE_CONV_2D' if n%2==0 else 'CONV_2D'
        assert op['kind'] == expected_kind
        if n: assert op['inputs'][0] == m['operators'][n-1]['outputs'][0]
        inp, out = t[op['inputs'][0]], t[op['outputs'][0]]
        kind = {'QUANTIZE': 0, 'CONV_2D': 1, 'DEPTHWISE_CONV_2D': 2,
                'MEAN': 3, 'FULLY_CONNECTED': 4}[op['kind']]
        ish, osh = inp['shape'], out['shape']
        ih, iw, ic = ish[1:] if len(ish) == 4 else (1, 1, ish[-1])
        oh, ow, oc = osh[1:] if len(osh) == 4 else (1, 1, osh[-1])
        src = 0 if n <= 1 else (1 if n % 2 == 0 else 2)
        dst = 0 if n == 0 else (1 if n % 2 else 2)
        assert oh * ow * oc <= [12288, 16384, 8192][dst]
        kh = kw = stride = 1
        wb = bb = 0
        qb = len(qparams)
        iz, oz = inp['zero_points'][0], out['zero_points'][0]
        lo, hi = -128, 127
        if kind in (1, 2, 4):
            w, b = (t[i] for i in op['inputs'][1:])
            assert all(z == 0 for z in w['zero_points'])
            if kind != 4:
                kh, kw = w['shape'][1:3]
                stride = op['StrideH']
                assert stride == op['StrideW'] and op['DilationHFactor'] == op['DilationWFactor'] == 1
                assert kind != 2 or op['DepthMultiplier'] == 1
            wb, bb = w['constant_offset'], b['constant_offset']
            weights = np.frombuffer(raw, dtype=np.int8, count=w['constant_bytes'], offset=wb)
            weights = weights.reshape(-1, oc).T if kind == 2 else weights.reshape(oc, -1)
            bias = np.frombuffer(raw, dtype='<i4', count=oc, offset=bb).astype(np.int64)
            bound = np.abs(bias) + np.abs(weights.astype(np.int64)).sum(axis=1) * max(abs(-128-iz), abs(127-iz))
            assert np.all(bound <= 2147483647), 'int32 partial sum could overflow'
            for c in range(oc):
                q, s = multiplier(inp['scales'][0] * w['scales'][c] / out['scales'][0])
                qparams.append(((s & 255) << 32) | q)
            if op['fused_activation'] == 'RELU6':
                lo = max(-128, oz)
                hi = min(127, int(math.floor(6 / out['scales'][0] + .5)) + oz)
            else:
                assert op['fused_activation'] == 'NONE'
        elif kind == 3:
            assert (ih, iw) == (2, 2) and inp['scales'] == out['scales'] and iz == oz == -128
        else:
            assert inp['scales'] == out['scales'] and iz - oz == 128
        py = max(0, (oh-1)*stride+kh-ih)//2 if kind in (1, 2) else 0
        px = max(0, (ow-1)*stride+kw-iw)//2 if kind in (1, 2) else 0
        assert all(v & (v-1)==0 for v in (iw,ic,ow,oc)) or n in (0,1)
        vals = [kind, src, dst, ih, iw, ic, oh, ow, oc, kh, kw, stride, py, px, iz, oz, wb, bb, qb, lo, hi,
                iw.bit_length()-1,ic.bit_length()-1,ow.bit_length()-1,oc.bit_length()-1]
        descriptors.append(dict(zip(fields.split(','), vals)))
        cases.append(f"{n}: begin " + ' '.join(f'{k} <= {v};' for k, v in zip(fields.split(','), vals)) + ' end')
    (dest / 'cnn_profile.vh').write_text('case (layer)\n' + '\n'.join(cases) + '\ndefault: kind <= 7;\nendcase\n')
    assert len(qparams) == 2740
    mem(dest / 'qparams.mem', qparams, 40)
    mem(dest / 'weights.mem', np.frombuffer(raw,dtype='<u4'), 32)
    # Explicit true-dual-port 4 KiB banks avoid ROM duplication for readback
    # and prevent synthesis rounding the whole payload up to a 256 KiB bank.
    banks=[]
    for bank in range((len(raw)+4095)//4096):
        block=raw[bank*4096:(bank+1)*4096].ljust(4096,b'\0')
        init=',\n'.join(f"    .INIT_{i:02X}(256'h{block[i*32:(i+1)*32][::-1].hex()})" for i in range(128))
        banks.append(f'''RAMB36E1 #(
    .RAM_MODE("TDP"),.SIM_DEVICE("7SERIES"),
    .READ_WIDTH_A(36),.READ_WIDTH_B(36),.WRITE_WIDTH_A(36),.WRITE_WIDTH_B(36),
    .DOA_REG(0),.DOB_REG(0),
{init}
) bank_{bank} (
    .CLKARDCLK(clk),.CLKBWRCLK(clk),
    .ADDRARDADDR({{1'b0,host_addr[11:2],5'b0}}),
    .ADDRBWRADDR({{1'b0,core_addr[11:2],5'b0}}),
    .ENARDEN(host_en && host_addr[17:12]==6'd{bank}),
    .ENBWREN(core_addr[17:12]==6'd{bank}),
    .DOADO(host_words[{bank}]),.DOBDO(core_words[{bank}]),
    .DOPADOP(),.DOPBDOP(),.DIADI(32'b0),.DIBDI(32'b0),.DIPADIP(4'b0),.DIPBDIP(4'b0),
    .WEA(4'b0),.WEBWE(8'b0),.REGCEAREGCE(1'b1),.REGCEB(1'b1),
    .RSTRAMARSTRAM(1'b0),.RSTRAMB(1'b0),.RSTREGARSTREG(1'b0),.RSTREGB(1'b0),
    .CASCADEINA(1'b0),.CASCADEINB(1'b0),.CASCADEOUTA(),.CASCADEOUTB(),
    .INJECTDBITERR(1'b0),.INJECTSBITERR(1'b0),.DBITERR(),.SBITERR(),.ECCPARITY(),.RDADDRECC()
);''')
    (dest/'cnn_weight_banks.vh').write_text('\n'.join(banks)+'\n')
    mem(dest / 'pixels.mem', np.asarray(m['input_pixel_lut'], dtype=np.int32)-128)
    (dest / 'profile.json').write_text(json.dumps(dict(model_sha256=m['model_sha256'], descriptors=descriptors,
        qparam_count=len(qparams), configuration='fixed ROM; rebuild bitstream to change profile'), indent=2)+'\n')
    return m, descriptors


def fixtures(m, images, cifar_test=None):
    import sys
    sys.path.insert(0, str(ROOT / 'software'))
    from animal_cnn.mobilenet_export import interpreter, pixel_lut
    from PIL import Image, ImageOps
    out = ROOT / 'build/fixtures'
    out.mkdir(parents=True, exist_ok=True)
    samples = [np.load(BUNDLE / 'trace.npz')['raw_rgb']]
    if not images and not cifar_test:
        images = [ROOT / 'sim/images' / f'{name}.png' for name in m['classes']]
    if cifar_test:
        import pickle
        # Only use the trusted CIFAR-10 Python distribution's test_batch file.
        with open(cifar_test, 'rb') as f:
            dataset = pickle.load(f, encoding='bytes')
        rgb = dataset[b'data'].reshape(-1,3,32,32).transpose(0,2,3,1)
        labels = np.asarray(dataset[b'labels'])
        for label in (2,3,5,7):
            index = int(np.where(labels == label)[0][0])
            samples.append(np.asarray(Image.fromarray(rgb[index]).resize(
                (64,64), Image.Resampling.BILINEAR), dtype=np.uint8)[None])
    for filename in images:
        with Image.open(filename) as im:
            samples.append(np.asarray(ImageOps.exif_transpose(im).convert('RGB').resize(
                (64, 64), Image.Resampling.BILINEAR), dtype=np.uint8)[None])
    runner = interpreter(BUNDLE / 'model.tflite', preserve=True)
    runner.allocate_tensors()
    inp = runner.get_input_details()[0]
    constants = np.frombuffer((BUNDLE / 'constants.bin').read_bytes(), dtype='<u4')
    mem(out / 'constants.mem', constants, 32)
    expected_scores = []
    for n, image in enumerate(samples):
        runner.set_tensor(inp['index'], pixel_lut(inp)[image])
        runner.invoke()
        mem(out / f'image_{n}.mem', np.frombuffer(image.tobytes(), dtype='<u4'), 32)
        flat = []
        for op in m['operators']:
            a = runner.get_tensor(op['outputs'][0])
            flat.extend(a.reshape(-1).tolist())
        mem(out / f'golden_{n}.mem', flat)
        expected_scores.append(flat[-4:])
    (out / 'count.txt').write_text(str(len(samples)))
    c_images=[]
    for image in samples:
        flat_image=image.reshape(-1).tolist()
        c_images.append('{\n'+',\n'.join(','.join(str(v) for v in flat_image[i:i+32]) for i in range(0,len(flat_image),32))+'\n}')
    (out/'cnn_test_images.h').write_text('#ifndef CNN_TEST_IMAGES_H\n#define CNN_TEST_IMAGES_H\n'
        +f'#define CNN_TEST_IMAGE_COUNT {len(samples)}U\n'
        +f'#define CNN_TEST_MODEL_CRC32 0x{zlib.crc32((BUNDLE/"constants.bin").read_bytes()):08x}U\n'
        +'static const signed char cnn_test_scores[CNN_TEST_IMAGE_COUNT][4] = {\n'
        +',\n'.join('{'+','.join(str(v) for v in scores)+'}' for scores in expected_scores)+'\n};\n'
        +'static const unsigned char cnn_test_images[CNN_TEST_IMAGE_COUNT][12288] = {\n'
        +',\n'.join(c_images)+'\n};\n#endif\n')
    (out / 'metadata.json').write_text(json.dumps(dict(model_sha256=m['model_sha256'],
        raw_rgb_sha256=[hashlib.sha256(s.tobytes()).hexdigest() for s in samples],
        tensor_bytes=len(flat),tensorflow='2.20.0', delegates=False), indent=2)+'\n')
    print(f'Compiled 30 operators; generated {len(samples)} image fixtures at {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--image', action='append', default=[])
    p.add_argument('--cifar-test', help='Trusted CIFAR-10 test_batch file; selects one image per deployed class')
    args = p.parse_args()
    manifest, _ = compile_profile()
    fixtures(manifest, args.image, args.cifar_test)
