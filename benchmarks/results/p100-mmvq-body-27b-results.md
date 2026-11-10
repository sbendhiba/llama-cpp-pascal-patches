# P100 MMVQ kernel-body adoption — Qwen3.8 27B results

Branch `p100-mmvq-body`, patches **90** (`mmvq-p100-body`) and **91** (`mmvq-f16-verify`), on top of the
76-patch `b12211` set. Both default ON with runtime kill switches (`GGML_CUDA_MMVQ_P100_BODY=0`,
`GGML_CUDA_MMVQ_F16=0`). Source: junkgpulab/llama.cpp-p100, adopted and re-gated.

**Arms.** `ours` = b12211-patched (the 76-patch set); `theirs` = junkgpulab build-p100 (base b11887);
`p91` = ours + patches 90+91. 3x Tesla P100-PCIE-16GB, 160 W each, `-sm tensor`, `-fa on -t 28 -ub 2048`,
cooled same-session A/B. Decode = llama-bench pp512+tg128, 2 reps. MTP = `--spec-type draft-mtp
--spec-draft-n-max 4 --spec-draft-p-min 0.2`, greedy, parallel 1, two prompts.

## Plain decode (3-card tensor), t/s

| quant | arm | tg128 | pp512 | tg vs ours |
|---|---|---:|---:|---:|
| Q4_K_M | ours | 34.88 | 490.79 | — |
| Q4_K_M | theirs | 33.90 | 379.83 | −2.8% |
| Q4_K_M | **p91** | **39.17** | **494.27** | **+12.3%** |
| Q6_K | ours | 29.34 | 509.26 | — |
| Q6_K | theirs | 31.20 | 397.39 | +6.3% |
| Q6_K | **p91** | **34.07** | **513.76** | **+16.1%** |

p91 wins decode outright on both quants — faster than ours and faster than theirs — with prefill
preserved (theirs sacrifices ~22% prefill). Raw: `final-27b-3card-160w-p100x3-{q4km,q6k}.csv`.

## MTP (3-card tensor), t/s (prompt0 / prompt1)

| quant | arm | p0 | p1 | vs ours |
|---|---|---:|---:|---:|
| Q4_K_M | ours | 58.87 | 30.53 | — |
| Q4_K_M | theirs | 79.07 | 49.26 | +34% / +61% |
| Q4_K_M | **p91** | **78.61** | **50.79** | **+33% / +66%** |
| Q6_K | ours | 55.13 | 34.44 | — |
| Q6_K | theirs | 71.38 | 44.04 | +29% / +28% |
| Q6_K | **p91** | **64.88** | **40.93** | **+18% / +19%** |

Q4_K_M MTP **matches** their fork. Q6_K MTP is +18% over ours with a ~9% residual to theirs (below).
Draft acceptance is unchanged across arms; zero CUDA errors. Raw: `final-mtp.txt`, `p90-mtp.txt`,
`p91-q6k.txt`.

## UD-IQ2_XXS (UD3.1) — patches apply globally (3-card tensor), t/s

| arm | tg128 | pp512 |
|---|---:|---:|
| stock (b12211-stock) | 23.46 | 271.50 |
| ours (76 patches) | 35.69 | 466.51 |
| theirs | 32.04 | 365.20 |
| **p91** | **38.45** | **469.00** |

stock→ours +52% decode; ours→p91 +7.7%; p91 beats theirs +20%. The 2-bit gain is smaller because IQ2_XXS
is bandwidth-bound. **MTP not measurable on this file**: the UD-IQ2_XXS GGUF carries no MTP layers
(llama-server refuses the MTP context); the Q4_K_M/Q6_K files do. Raw: `iq2xxs-check.txt`,
`iq2xxs-stock.txt`, `iq2xxs-27b-3card-*.csv`.

## Gates (all pass)

- Clean-room apply of the full 78-patch set at `-F0`: zero fuzz, zero offset, tree byte-identical.
- Both kernels build at `sm_60`, CUDA 12.4, NCCL off, no duplicate symbols.
- Per-op ADD/MUL_MAT/MUL_MAT_ID identical to ours (168/1131/932, 0 fail; the shared pre-existing q4_K
  `convert_func` abort unchanged, only its line number shifts by the added include).
- Logits determinism 4 PASS (verify repeated 3x identical, graphs-off identical, prefill 2x identical,
  graphs-off identical). Patch 91's fp16 verify shifts verify PPL 8.7717→8.8461 (NMSE ~5e-7), prefill
  unchanged; fully repeatable.
- Runtime flags verified: `GGML_CUDA_MMVQ_P100_BODY=0` reproduces ours; `GGML_CUDA_MMVQ_F16=0` falls back
  to the integer verify path.

## The remaining ~9% (Q6_K MTP only)

p91's Q6_K verify is slower than theirs both integer (56.3 vs 67.7) and fp16 (64.9 vs 71.4), even though
the mmvq and fp16 kernels are copied verbatim — so the carrier is a shared part of the verify step, not
the matvec kernel. Ruled out by measurement: the FFN GLU/GDN-gate fusions (~1%), the `mmv-p100` framework
(`GGML_CUDA_P100=1`, default off), the P60 block (no effect on the 27B), and `fattn-gemm` (needs Q≥128 /
KV≥4096, i.e. long-context prefill). Leading unconfirmed candidate is their `fattn.cu` verify-attention
dispatch, but its q4_0-cache kernels do not fire on the default f16 KV cache used here. nvprof aborts
(signal 134) on this CUDA 12.4 build, so the carrier is not yet pinpointed. Tracked as the next candidate.

An apparent decode "regression" during measurement was cross-session clock drift, not code: ours itself
read 29.40 in one clean session vs 23.74 in another, and the relative gain is +16–18% in every session.