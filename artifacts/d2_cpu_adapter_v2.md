# D2 CPU source adapter — 2026-09-08

Implemented a separate VNNI source entry point; the historical unsigned-byte
kernel and its results remain unchanged. The adapter preserves E4M3 codes as
qA=512*A and E2M1 weights as qW=2*W. It offsets qA by 2^18, uses three unsigned
byte planes, reconstructs the K32 integer dot and subtracts the weight-sum
compensation before applying source block scales and ordered FP32 reduction.
The kernel is compiled without floating-point contraction.

`OMP_NUM_THREADS=2 .venv-cu/bin/python -m mosaicvm.bench.d2_cpu_expert`
completed successfully. All 254 finite E4M3 codes are covered in bounded
integer-oracle tests, including signed/fractional/extreme values. Complete
L41 expert 0 GU/SwiGLU/route-weight/DOWN BF16 outputs match the ordered CPU
reference bitwise on the seeded synthetic row. The unchanged checkpoint
Expert AST controls clamp, route placement and cast boundaries.

Evidence: `d2_cpu_expert_v2.json` and `.log`. This is exact integer encoding
plus bit-equal tested CPU-reference cases, not global native-source bitwise
certification, full routed/shared tail validation, or production D2 latency.
Scale-byte 0/255 remains rejected by this adapter pending adapter-specific
edge handling. It does not change prior target-side semantics certification.

The prepared representation includes int8 weights, FP32 scales AND int32
per-block weight sums: 31,457,280 bytes for one expert. Fully expanding all
512 routed tail experts in this layout would require 16,106,127,360 bytes,
excluding canonical packed copies, head, state, workspace and allocator.
This is a derived footprint, not a measured full-tail resident allocation.

Next: source GPU parity and real routed/shared reduction, then complete-tail
latency/residency and contention. Do not reuse the old v4-B throughput figures
for this three-plane adapter.
