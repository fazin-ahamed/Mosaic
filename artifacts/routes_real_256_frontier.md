# Real ≥256-Position Route Capture Frontier (Blackwell sandbox)

Generated: 2026-08-29T12:38:48+0000
Model: deepseek-ai/DeepSeek-V4-Flash-0731 @ 7872f01b1d1fe23eabc4c98b48bffcef5a386062
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (95.0 GiB)

Classification: **BLACKWELL_SANDBOX_ONLY** — model-property capture,
NOT RTX 3060 performance evidence.

Captures: 7 prompts × 260 positions (batched greedy decode)
Total records: 78260 (260 consecutive positions per capture × 43 layers)
Validation: **PASS**

## Per-Layer Union Size (U) by Width — the frontier metric

Window-union saturates at 256 (16k assignments always touch every
expert). The meaningful metric is per-LAYER unique experts within
a window — comparable to historical U_med 86 (K56) / 91 (K63).

| Width | median-of-layer p50 | median-of-layer p90 |
|---:|---:|---:|
| 64 | 86 | 99 |
| 80 | 95 | 111 |
| 96 | 104 | 120 |
| 112 | 112 | 127 |
| 124 | 116 | 133 |
| 128 | 117 | 134 |
| 144 | 122 | 139 |
| 160 | 127 | 145 |

## Per-Layer Union Detail (p50 per layer at N160)

| Layer | p50 U | p90 U | mean U |
|---:|---:|---:|---:|
| 0 | 214 | 239 | 208.6 |
| 1 | 218 | 242 | 211.1 |
| 2 | 215 | 240 | 207.7 |
| 3 | 138 | 158 | 139.3 |
| 4 | 126 | 141 | 121.0 |
| 5 | 130 | 142 | 130.8 |
| 6 | 127 | 140 | 122.8 |
| 7 | 117 | 147 | 122.5 |
| 8 | 114 | 133 | 116.4 |
| 9 | 138 | 159 | 139.1 |
| 10 | 141 | 158 | 142.8 |
| 11 | 127 | 138 | 124.5 |
| 12 | 106 | 121 | 102.9 |
| 13 | 109 | 119 | 107.2 |
| 14 | 108 | 117 | 103.0 |
| 15 | 107 | 127 | 107.6 |
| 16 | 119 | 135 | 116.6 |
| 17 | 122 | 135 | 119.1 |
| 18 | 118 | 144 | 118.7 |
| 19 | 150 | 171 | 146.0 |
| 20 | 131 | 154 | 132.2 |
| 21 | 143 | 155 | 137.5 |
| 22 | 140 | 154 | 136.7 |
| 23 | 132 | 144 | 129.0 |
| 24 | 103 | 118 | 102.9 |
| 25 | 120 | 130 | 114.6 |
| 26 | 113 | 122 | 110.6 |
| 27 | 130 | 151 | 130.3 |
| 28 | 122 | 140 | 121.7 |
| 29 | 124 | 144 | 123.4 |
| 30 | 127 | 147 | 127.5 |
| 31 | 141 | 154 | 141.5 |
| 32 | 124 | 148 | 123.2 |
| 33 | 126 | 145 | 125.7 |
| 34 | 119 | 128 | 118.8 |
| 35 | 158 | 172 | 153.7 |
| 36 | 116 | 140 | 117.6 |
| 37 | 139 | 156 | 133.8 |
| 38 | 123 | 147 | 120.9 |
| 39 | 149 | 176 | 144.6 |
| 40 | 123 | 138 | 117.0 |
| 41 | 133 | 160 | 135.3 |
| 42 | 134 | 150 | 130.0 |

## Expert Frequency

- Total assignments: 469560
- Unique experts used: 256 / 256
- Top-10 share: 6.4%
- Top-20 share: 11.7%
- Per-layer Gini: median 0.5575 (min 0.2949, max 0.6376)

## Neighboring-Position Overlap

- Mean Jaccard (consecutive positions, same layer): 0.235

## Notes

- Genuine consecutive positions; no corpus wrapping
- Greedy deterministic batched decode (left-padded prompts)
- Raw records preserved: routes_real_256_raw.json (19.4 MB)
- BranchPack conclusions downstream; raw routes remain scheduler-agnostic
- Overlap corpus for 3060 numerical-equivalence check: routes_overlap_corpus_3060.json
- Model-property evidence only; all timing = BLACKWELL_SANDBOX_ONLY