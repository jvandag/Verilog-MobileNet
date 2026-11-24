# Fixed-profile Wishbone MobileNet

The implementation is in `rtl/`: `wb_cnn.sv`, `cnn_core.sv`, `cnn_ram.sv`, and
`cnn_weights.sv`.
It implements the 30-operator integer graph using synchronous memories and a
sequential MAC. `tools/compile_hardware.py` generates the profile include,
per-channel Q31 ROM, pixel LUT, diagnostic `weights.mem`, and hard-coded BRAM
initialization parameters in `cnn_weight_banks.vh` under `rtl/generated/`.

This release compiles both weights/biases and graph configuration into the bitstream.
`CONFIG_BYTES` reads zero; CFG_ADDR/CFG_DATA are unsupported. To retrain, rerun
the compiler and rebuild the bitstream with the corresponding constants.
Runtime weight loading, descriptor execution and descriptor validation are not implemented.
This keeps the memory schedule and buffer bounds fixed and reviewable.

## Reproduce image simulation

From this submodule, using Python 3.11 with the pinned runtime dependencies:

```powershell
.venv/Scripts/python tools/compile_hardware.py --image path/to/bird.jpg --image path/to/cat.jpg
$env:VIVADO_BIN='E:/AMDDesignTools/2026.1/Vivado/bin'
powershell -NoProfile -ExecutionPolicy Bypass -File sim/run.ps1 -Count 3
```

Default compilation includes the supplied raw image and four bundled CIFAR-10
test photographs in `sim/images/`. The supplied raw image is always fixture zero.
`--image` accepts repeated real
image paths, applies EXIF orientation/RGB/PIL bilinear 64x64 resize, and runs the
native TensorFlow 2.20.0 interpreter without delegates. Alternatively,
`--cifar-test path/to/test_batch` selects the first held-out bird/cat/dog/horse
images from a trusted local CIFAR-10 Python distribution. Do not unpickle an
untrusted file. Generated fixtures, metadata and logs are under ignored `build/`.

The self-checking xsim testbench reads back all 54,898 initialized ROM words
through Wishbone, checks ROM write protection, writes raw RGB through the IMAGE window,
starts inference, compares every intermediate byte in NHWC order, and verifies
final scores and sticky interrupts. It also checks masked writes, cancelled
requests, START without a complete image and busy protection. Mismatches stop at the first
image/layer/address with expected and actual bytes. Add each failing image to
the regression invocation before fixing RTL; regenerate goldens only from the
software oracle. Class agreement alone is not acceptance.

The runner first tests **both ports of the actual 54 RAMB36E1 banks** against all
54,898 expected words, including bank boundaries and host read-enable behavior.
Long image inference then uses a one-cycle, cycle-equivalent ROM model selected
only by `CNN_FAST_SIM`. Pass `-PrimitiveInference` to run the complete image test
with the much slower Xilinx primitive models instead. Synthesis always uses the
banked primitives, not the simulation model. The weights occupy BRAM, not a
flip-flop register array, and are ready immediately after configuration/reset.

## Arithmetic

The tested Windows TensorFlow 2.20.0 native CPU kernels use a single rounded Q31
product for convolution/dense and double rounding for depthwise. The MEAN kernel
also uses double rounding: for this fixed four-sample, equal-scale graph its
centered unsigned sum is equivalent to `(sum + 3) >> 2`, then subtracting 128.
These distinctions were verified against full intermediate traces. The signed
MAC path widens centered inputs to nine bits and accumulates into int32. The
compiler bounds partial sums using int64 and rejects positive requantizer shifts
or unsupported graph options. No floating-point arithmetic exists in RTL.

## Host contract

Use the MMIO offsets in `hardware-plan.md`, with the fixed-ROM exception above.
Read ID `0x434e4e33`. MODEL_VALID is set immediately after reset because weights,
biases, requantization and the graph are built in. No program needs to load or
commit weights. MODEL_COMMIT (4) is retained as a harmless compatibility command.
PARAM_ADDR/PARAM_DATA support indirect ROM readback; PARAM_DATA writes are
rejected with error 7 and do not change MODEL_VALID or the ROM. Replacing weights
requires recompiling the profile and rebuilding/programming the bitstream.

For each image, write all 3,072 IMAGE words in ascending order, clear old causes
through IRQ_STATUS (W1C), then START (1). Image byte enables work,
but the completeness counters require ascending full-word loading; partial or
out-of-order image writes require restarting the complete image load.
Image preprocessing is in place: always reload raw RGB before another START.
Poll STATUS or enable DONE/ERROR IRQ bits. Read scores only after DONE; they are
signed int8 values sign-extended to 32 bits, not probabilities. ABORT (2) preserves
the built-in model but invalidates the image. Bus errors ACK and set sticky
ERROR without interrupting an active inference. Clear ERROR before restarting.

Error codes: 1 invalid access, 2 invalid command, 3 busy/protected, 4 reserved for
missing model, 5 stale causes or incomplete image, 6 parameter bounds, 7 ROM write.
Completion events take priority over W1C. Reset clears control state without
erasing BRAM. A direct Wishbone master must keep CYC asserted through ACK.
The wrapper captures address/data/controls on STB. The parent mux now retains
the selected request through ACK and asserts STALL to limit its pipelined
AXI/Wishbone bridge to one outstanding peripheral request.

## Synthesis and integration

`tools/synth.tcl` synthesizes the peripheral alone for XC7A100T-CSG324-1 and
records utilization/timing under `build/synth/`. This is not a routed timing
result. The parent repository's `scripts/cnn_system.tcl` builds the integrated
system, adds region 5 and interrupt bit 6, and exports an XSA only after
implementation. See the parent implementation report for measured acceptance.
