# M=32 standalone projection memory core

Validated on the PYNQ-Z2 at 100 MHz: see [실측 결과](RESULTS.md).

This project is a SW–HW co-design prototype for projecting window textures into
the BOSIO M=32 triangular scene layout. It does not instantiate the Output Core,
HDMI, or sensor logic. The first milestone is verified DDR contents, not 60 Hz
full-sphere composition.

## Work split

- **CPU:** window basis/FOV, tile geometry, fixed-point coefficient preparation,
  conservative tile culling, layer ordering, buffer allocation and cache maintenance.
- **FPGA:** 1,024 triangular cell coordinates per job, perspective division,
  four-texel bilinear sampling, RGB332 packing, tile read/modify/write in DDR.
- No EULSUKDO scheduler or full-sphere direction-vector table is used.

The implementation is synthesizable C++ using Vitis HLS 2022.2; generated Verilog
is validated by C/RTL co-simulation, then implemented with Vivado 2022.2 for
`xc7z020clg400-1` at 100 MHz. The arithmetic cell loop is sequential. It is **not**
a one-cell-per-clock pipeline and has no texture cache in this first version.

## Data contract

Texture words are `0x00RRGGBB`; texture offset and stride are in 32-bit words.
The output is four RGB332 indices per little-endian word, matching the existing
Output Core cell payload ordering. A tile always occupies 256 words / 1,024 bytes.
Scene palette, tile directory and frame-boundary ownership are CPU responsibilities;
the standalone test writes only cell payloads.

Each 18-word descriptor contains:

| Word | Meaning |
|---:|---|
| 0 | Texture base offset in words |
| 1 | Destination tile offset in words |
| 2–3 | Texture width/height, 1–2048 |
| 4 | Texture row stride in words, at least width |
| 5 | bit 0 clear tile before drawing; bits 15:8 RGB332 background |
| 6–9 | U coefficients: base, row, column, inverted-cell delta |
| 10–13 | V coefficients, same order |
| 14–17 | D coefficients, same order |

Coefficients are signed 32-bit integers with one common scale. For row `r`, column
`c`, orientation `i`: `U=U0+r*Ur+c*Uc+i*Ui`, likewise V/D. Coordinate division
uses `floor(256*U/D)` and `floor(256*V/D)`. Cells with `D<=0`, negative numerators,
or coordinates outside `[0,width-1] × [0,height-1]` leave the tile background or
prior value intact. Interpolation uses 8 fractional bits and round-to-nearest
channel output before RGB332 quantization. It is not bit-identical to the old
floating-point compositor's AA or arbitrary window shapes.

`abs(base) + 31*abs(row) + 31*abs(column) + abs(inverted)` is bounded using absolute
values for all four terms to prevent arithmetic overflow. Invalid dimensions,
offsets, flags or coefficients are rejected before that descriptor touches the
destination. Completed earlier descriptors remain committed if a later job fails.
Input and output allocations must be separate. All pointers and bounds come from
owned `pynq.allocate` buffers; passing an arbitrary physical address is unsupported.

Return codes: 0 success; `0x10000000` job/descriptor count error;
`0x20000000 | job` texture/destination/format error;
`0x30000000 | job` coefficient range error. `written_cells` counts sampled cells,
including repeated writes by multiple ordered jobs; it is not a unique-cell count.

## Build and verify

1. Run `scripts/build_hls.tcl` with Vitis HLS 2022.2. C simulation and RTL
   co-simulation must pass before IP export.
2. Run `scripts/build_overlay.tcl` with Vivado 2022.2. This creates a dedicated
   PS + AXI + projection overlay and exports bit/hwh only after timing passes.
3. Run `python sw/test_model.py` for geometry and conservative culling checks.
4. Generate `sw/registers.json` from the HLS driver header (current map is saved).
5. Copy the repository to an isolated directory on PYNQ and run
   `sw/board_validate.py` in the PYNQ Python environment after stopping display
   services. This test loads a standalone overlay; do not run it while the live
   compositor owns the projector engines.

The combined overlay and cached compositor are in
[bosio_SphericalWM](https://github.com/VARZero/bosio_SphericalWM/tree/main/overlays/dual_projection).
`sw/dual_projector.py` provides the tested dual-engine PS driver; `partition_jobs.py`
preserves per-tile layer order and assigns disjoint destinations. Run
`python sw/test_partition.py` to check scheduling.

The test compares the entire output allocation, including untouched regions and
guards, against an independent vectorized fixed-point model; checks input/descriptor
immutability; exercises clipping, bilinear edges, clear/preserve, invalid bounds,
full-sphere and culled jobs, and ordered overlay jobs. Buffers are released only
after the accelerator completes. A timeout retains them until the core is idle.
The validator explicitly sets and reads back FCLK0=100 MHz; the initial clock
inherited from the overlay may differ from the routed timing target.
Reported job milliseconds measure AXI start-to-done with Python polling and exclude
CPU descriptor preparation, cache flush/invalidate, and reference-model time.

`sw/geometry_reference.py` is an unchanged copy of the existing OutputCore geometry
module. `test_model.py` also compares tile vertices against that module and projected
coordinates against direct 3D dot/divide calculations. Build and board results are
saved in `results/`; source files and deployed artifacts have SHA-256 manifests.

## Deferred work

Texture caching, overlapped/pipelined divide and sampling, alpha/coverage AA,
and arbitrary shape masks remain future stages. Concurrent Output Core integration
is implemented in the separate SphericalWM repository.
Do not use this prototype's resource count as the cost of a one-cell-per-clock
compositor, or its memory-test throughput as HDMI refresh rate.
