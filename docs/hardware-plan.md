# Int8 MobileNet SystemVerilog implementation plan

## Scope and current status

Implement the sole supported profile: **MobileNetV1-0.25, 64x64 RGB, four outputs**
(bird, cat, dog, horse), with int8 weights/activations, int32 biases/accumulators,
and integer requantization. Preserve the supplied integer model and
[numeric contract](numeric-contract.md). The measured software test accuracy is
82.05%. The fixed-profile RTL, compiler and image testbench are now implemented;
see [RTL implementation](rtl-implementation.md) for the executable host contract
and deviations from this original plan. The parent implementation report records
measured synthesis, routed timing and board acceptance separately.

The implemented weight policy is **immutable, initialized FPGA BRAM ROM**.
Weights and biases are part of the bitstream; MODEL_VALID is set after reset
and software uploads only images. PARAM_ADDR/PARAM_DATA provide readback;
PARAM_DATA writes are rejected. This supersedes the runtime parameter-loading
and commit proposals below. Retraining requires regenerating ROM initialization
and rebuilding the bitstream.

The graph and payload are documented in [mobilenet.md](mobilenet.md). Pin the
release hashes there before implementation. Retraining the same architecture
requires recompiling its quantization configuration, not just replacing weights.

## Parent integration

Findings are pinned to [ECE544_Final_Project commit
9e09ced7b4a84a02e0d0c95455dc1f46c7de022b](https://github.com/CaseyRasor/ECE544_Final_Project/tree/9e09ced7b4a84a02e0d0c95455dc1f46c7de022b).
Recheck them when integrating into a newer parent revision.

| Parent evidence | Integration requirement |
| --- | --- |
| `final_project.xpr` | XC7A100T-CSG324-1, Nexys A7-100T |
| `hdl/mbc_wb_nexys4DDR.sv` | 100 MHz; word-address bridge output converted to byte addresses with `{4'b0,m_wb_adr,2'b00}` |
| `hdl/wishbone/wb_mux.sv` | Byte-address decode `[18:16]`; regions 5-7 unused, each 64 KiB |
| `mbc_wb_embsys.bd` | Existing AXI aperture 512 KiB at 0x80000000; reserve region 5 at **0x80050000** |
| Existing Wishbone peripherals | One-cycle STB; CYC remains high until ACK; one outstanding transaction |
| Top-level bus wiring | STALL and ERR tied off; respond to invalid accesses with ACK and a sticky error |
| Interrupt controller/concat | All six current IRQ inputs used; custom IRQs currently edge-sensitive |

Expose `i_clk`, active-high synchronous `i_reset`, Wishbone CYC/STB/WE,
32-bit byte address/data, four byte SELs, output ACK/data, and `o_irq`.
Capture the entire request on `CYC && STB` when idle. Complete registered BRAM
reads using the latched address/controls even after STB falls. ACK exactly once;
cancel pending requests if CYC drops or reset occurs. Respect byte enables.
No access may wait for a classification to finish; busy/protected accesses must
ACK promptly and report an error. Add a region-5 mux slave and corresponding bus
wiring; the existing AXI aperture already covers it.

Add a seventh interrupt-controller input (bit 6) and a new Wishbone IRQ route.
Use **level-high** sensitivity for CNN: change the pinned `C_KIND_OF_INTR` from
`0xFFFFFFFC` to `0xFFFFFFBC`, preserving the other inputs. Regenerate the block
design, wrapper, XSA and BSP; add the CNN base address/IRQ to software definitions.

## Shared compute engine

Use one time-multiplexed MAC lane initially, not an unrolled network. Reuse its
multiplier and accumulator across standard 3x3, pointwise 1x1, depthwise 3x3,
and dense operations. A sequencer traverses spatial positions/channels, supplies
padding values equal to each activation zero point, and schedules synchronous
BRAM reads. Depthwise mode connects each output channel to its matching input
channel. Fuse bias and ReLU6 with the requantization stage. Include integer input
conversion, global average pooling and four-way argmax.

Suggested modules: `wb_cnn.sv` (registers/bus/IRQ), `cnn_sequencer.sv`,
`cnn_mac.sv`, `cnn_requantize.sv`, `cnn_pool.sv`, and BRAM wrappers. These filenames
are implementation tasks, not existing modules. Pipeline the requantizer where
needed; a 32x32-to-64-bit rescale multiplication can consume extra DSP slices or
be serialized. The dot-product lane alone does not describe total DSP usage.

At 100 MHz, 3,329,536 MACs imply a **33.30 ms arithmetic lower bound** at one
MAC/cycle; three cycles per MAC imply 99.89 ms before conversion, requantization,
pooling, control and MMIO overhead. Measure cycle counts in RTL; neither number
is a promised frame rate.

## Memory and fit budget

The 100T has 63,400 LUTs, 126,800 flip-flops, 240 DSP48E1 slices and 135 RAMB36
blocks ([Digilent board reference](https://digilent.com/reference/programmable-logic/nexys-a7/start)).
The raw 4,860 Kibit BRAM figure includes parity capacity. Budget ordinary
32-bit/byte storage as approximately 4,096 usable bytes per RAMB36, then verify
actual inferred RAM modes and packing in Vivado.

| Memory | Size | Planned RAMB36 blocks |
| --- | ---: | ---: |
| Constant payload (weights/biases/axes) | 219,592 bytes | 54 |
| Raw/input image | 12,288 bytes | 3 |
| Activation A | 16,384 bytes | 4 |
| Activation B | 8,192 bytes | 2 |
| Compiled per-channel configuration, descriptors, LUT | Up to 16 KiB | 4 |
| Total before implementation packing changes | | **67** |

A first configuration budget is 2,740 channel multipliers (4 bytes each), 2,740
signed shifts (1 byte each), 30 descriptors (64 bytes each), and a 256-byte pixel
LUT: 15,876 bytes. The compiler must prove that its final encoding fits. Parameter
RAM needs at least 18 address bits; read offsets come from the manifest.

Perform both pixel-LUT mapping and the initial QUANTIZE **in place in IMAGE**.
Then conv1 writes A (8 KiB), depthwise1 writes B (8 KiB), and pointwise1 writes
A (16 KiB). Continue alternating A/B; prove all subsequent tensor lifetimes and
sizes. Putting the initial QUANTIZE in a ping-pong activation bank changes this
parity and can overrun the 8 KiB bank. Never overwrite live source activations.

**Planning estimates for the complete CNN peripheral:** 2,000-6,000 LUTs,
2-6 DSPs, and roughly 64-72 RAMB36 blocks, depending on memory packing and
requantizer implementation. These are design estimates, not synthesis results.
BRAM is the likely limit even if 80% of logic fabric is available. The inspected
parent has no checked-in utilization report. Its approximately 256 KiB processor
memory alone could consume about 64 RAMB36s, leaving little space after the CNN
and other peripherals. Confirm this against actual synthesis, not the logic
percentage assumption. Avoid duplicating the entire weights as a C array in
processor BRAM; stream from a host or external storage. If the combined design
does not fit, evaluate external DDR or processor-memory changes before scaling
MAC parallelism. The board's DDR is not an already-integrated CNN memory path.

## Proposed byte-addressed MMIO map

All offsets below are relative to 0x80050000. The payload exceeds a 64 KiB slave
region, so load it through indirect ports rather than a direct parameter window.

| Offset | Register/window | Meaning |
| --- | --- | --- |
| 0x0000 | ID (RO) | Profile ID `0x434E4E33` (CNN3) |
| 0x0004 | COMMAND (WO) | START bit 0, ABORT bit 1, MODEL_COMMIT bit 2; one command per write |
| 0x0008 | STATUS (RO) | BUSY bit 0, DONE bit 1, ERROR bit 2, MODEL_VALID bit 3 |
| 0x000C | IRQ_ENABLE (RW) | DONE bit 0, ERROR bit 1 |
| 0x0010 | IRQ_STATUS (RW1C) | Sticky DONE bit 0, ERROR bit 1 |
| 0x0014 | CLASS_ID (RO) | Lowest-index argmax of final quantized scores |
| 0x0018 | CYCLES (RO) | Inference cycle count, latched on completion |
| 0x001C | ERROR_CODE (RO) | Invalid address/command, busy, no model, bounds/configuration fault |
| 0x0020 | CONSTANT_BYTES (RO) | 219,592 for supplied profile |
| 0x0024 | INPUT_BYTES (RO) | 12,288 |
| 0x0028 | NUM_CLASSES (RO) | 4 |
| 0x002C | CONFIG_BYTES (RO) | Compiler-defined configuration length |
| 0x0030/34 | PARAM_ADDR / PARAM_DATA | Aligned byte offset / auto-incrementing 32-bit constant access |
| 0x0038/3C | CFG_ADDR / CFG_DATA | Aligned byte offset / auto-incrementing configuration access |
| 0x0100-0x010C | LOGITS[0..3] (RO) | Final int8 scores sign extended to 32 bits |
| 0x1000-0x3FFF | IMAGE (RW) | Raw 64x64x3 RGB bytes |

Indirect addresses increment by four only when a valid DATA request completes.
Check bounds/alignment before every access and honor SEL. Reading constants back
is supported for host verification. Writes to parameters/configuration invalidate
MODEL_VALID. Reject IMAGE or parameter/configuration writes while BUSY. Before
commit, software must load and verify the complete matching constants and
configuration, including hashes/readback; hardware does not implement SHA256.
MODEL_COMMIT checks fixed-profile descriptor sizes, tensor ranges, operation
counts and supported options before setting MODEL_VALID. A committed image must
not reference addresses outside allocated memories.

START requires idle, MODEL_VALID, and cleared prior DONE/ERROR status. It clears
the cycle counter, locks relevant memories, and begins input conversion. Publish
class, all scores and cycles atomically before setting sticky DONE. Faults stop
the engine and set sticky ERROR with an error code. ABORT flushes the pipeline,
clears BUSY/DONE, and preserves committed parameters; it does not publish a
partial result. Reset clears pending bus/IRQ/valid state without requiring BRAM
contents to be erased. Define `o_irq = |(IRQ_STATUS & IRQ_ENABLE)`. Software
clears causes using W1C; a simultaneous new event wins over a clear. STATUS
DONE/ERROR mirror the corresponding IRQ_STATUS flags. Stale output registers
must only be consumed after DONE.

## Software exporter/compiler work

The existing exporter produces TFLite, raw constants, operator/tensor metadata,
and golden intermediate vectors. **It does not yet produce executable hardware
configuration.** Add a compiler which:

1. Validates the fixed graph, layouts, calibration metadata and checksums.
2. Folds/checks zero-point corrections and bounds every int32 partial sum.
3. Generates TFLite-compatible Q31 multipliers/shifts, ReLU6 clamps, input
   conversion and MEAN rules; tests signed rounding and saturation corners.
4. Assigns IMAGE/A/B buffers and proves lifetimes/capacity; emits binary op
   descriptors and integer configuration, with a versioned header and size checks.
5. Emits numeric little-endian 32-bit C/MMIO words and SystemVerilog packages,
   plus raw RGB and expected per-layer vectors. Constants require 54,898 words;
   firmware writes numeric words via `Xil_Out32`, respecting its byte order.

## Implementation and acceptance order

1. Synthesize the parent alone and record LUT/FF/DSP/BRAM usage and timing.
   Establish the combined memory budget before building the complete engine.
2. Build/test the hardware configuration compiler against the supplied integer
   reference. Require exact integer layer outputs, including input conversion,
   padding, depthwise, requantization, MEAN, dense and final tie behavior.
3. Implement the sequential engine and directed MAC/requantizer tests, then
   compare every layer of multiple images against `trace.npz`/new golden traces.
4. Add Wishbone registers and test pulsed STB, delayed reads, held CYC, byte SEL,
   reset/cancel, bad bounds, busy accesses, commit, abort, sticky IRQ and W1C races.
5. Integrate mux, level IRQ and firmware. Load/verify constants and configuration,
   write IMAGE, START, wait for interrupt, read class/scores and W1C acknowledge.
6. Synthesize/implement the combined project; require 100 MHz timing closure and
   real resource fit. Compare board scores with software for a representative
   corpus before reporting hardware accuracy or speed. Keep a polled-mode test
   for debugging, then validate interrupt delivery and repeated classifications.

Completion means bit-exact inference, reliable bus/interrupt behavior, measured
fit/timing and on-board validation. The software's measured accuracy alone does
not establish any of those hardware results.
