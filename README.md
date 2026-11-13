# llama-cpp-pascal-patches

Performance patches for [llama.cpp](https://github.com/ggml-org/llama.cpp), developed and measured on
Tesla P100 (GP100, `sm_60`). Not a fork: one `.patch` file per change, applied in numeric order with zero
fuzz, each carrying its own measurement, its kill switch and its dead ends in the comments it adds to the
source. Unofficial, unaffiliated with the llama.cpp project, MIT.

GP100 is the reason any of this exists. Alone among Pascal chips it has no `DP4A` and has full-rate
`HFMA2`, so llama.cpp's quantized matmul paths fall back to int emulation and cuBLAS while the one
instruction the card is genuinely good at goes unused. The rest is what the last two years of llama.cpp
made possible: a fused norm, deferred gathers, CUDA graphs, speculative decoding with an MTP head.

`main` is the **`v1.0.0`** set - **79 patches** against llama.cpp `b1b770d` (v1.0.0, 148 commits past
`b12211`): 39 we carry, the 37-patch **P60 block** (52-89, a port and retune of
[Mrinohk/qwen36-p100-patches](https://github.com/Mrinohk/qwen36-p100-patches) and
[qr0n/p100-llama-cpp](https://github.com/qr0n/p100-llama-cpp) work for the same cards) and **90-92**,
adopted from [junkgpulab/llama.cpp-p100](https://github.com/junkgpulab/llama.cpp-p100). Nothing was
dropped at this ref and nothing new was needed: v1.0.0 absorbed no patch here, so the whole set is the same
79 files re-anchored. Two went at the previous ref because upstream made them unnecessary: **59** (the MoE
expert-tile rule is now unconditional in `mmq.cu`) and **32** (upstream added its own split-input prefetch,
same field names). A llama.cpp **release** is maintained on `main`; a llama.cpp **pre-release build** gets a
branch of its own (`b12211` is one, `b11539` and `b11601` were), which is why `git tag -l` and `git branch`
answer different questions - see [RELEASING.md](RELEASING.md).

## Speed

3x Tesla P100-PCIE-16GB at **160 W**, CUDA 12.4, `-fa on -t 28 -ub 2048`, llama-bench, three cards
throughout at this ref, all arms in one session so a release delta and a port delta are not read out of two
different afternoons ([docs/results.md](docs/results.md)).

| model | split | stock `v1.0.0` pp512 / tg128 | this set pp512 / tg128 |
|---|---|---:|---:|
| Cyber-Tiel-Coder-35B-A3B MTP UD-Q4_K_XL (MoE) | `-sm layer` | 657.79 / 58.22 | **1081.02 / 118.32** (+64% / +103%) |
| | `-sm tensor` | 903.39 / 38.83 | **1618.13 / 80.36** (+79% / +107%) |
| Qwen3.8-27B UD-Q4_K_M (dense) | `-sm tensor` | 283.64 / 23.08 | **497.77 / 39.54** (+75% / +71%) |
| | `-sm layer` | 145.69 / 13.34 | **209.48 / 22.45** (+44% / +68%) |

The MoE tensor `tg128` is the one row whose percentage is not worth taking seriously: that stock arm came
back 38.83 with a 5.29 spread across its three rounds, so +107% is a ratio against a moving baseline. The
patched side and the `pp512` column are 1.7 or better on every cell.

These rows are three-card rows; the `b12211` table was two cards unless stated, so they are not a re-take
of those numbers and should not be read as one. The port check that *is* apples to apples was run in one
block with both patched arms in it: `b12211`-patched 52.02 t/s against `v1.0.0`-patched 52.29 t/s on the
same four prompts, **+0.5%** - same prompts, same cards, same afternoon, only the build under test
different.

### Qwen3.8-27B with the MTP head

`llama-server` `/completion` at temperature 0 with `ignore_eos`, `--spec-type draft-mtp
--spec-draft-n-max 4 --spec-draft-p-min 0.2`, 3-card tensor, 4 prompts x 256 tokens x 2 reps x **both arm
orders**, from [benchmarks/mtp-ab.sh](benchmarks/mtp-ab.sh) against the committed
[prompt set](benchmarks/mtp-prompts.txt). Every arm is identity-checked (own port, own pid, own log naming
the model) so a request cannot be answered by the previous build's server. Cycles/second is derived from
`draft_n` and `draft_n_accepted` - llama.cpp v1.0.0 publishes those, not a draft length - and takes
speculative acceptance out of the number: tokens per cycle came out 3.40 on stock and 3.30 on this set, so
the throughput gain is not an acceptance accident in either direction.

| UD-Q6_K_XL, 27B | stock `v1.0.0` | **this set (79)** | gain |
|---|---:|---:|---:|
| MTP, mean over 4 prompts | 36.29 t/s | **52.26** | **+44.0%** |
| MTP per decode cycle | 10.74 | **15.91** | +48.2% |
| MTP, p0 code | 39.85 | **57.72** | +44.7% |
| MTP, p1 prose / p2 haiku / p3 translation | 27.75 / 38.17 / 39.36 | **40.04 / 55.73 / 55.59** | +44.3 / +46.0 / +41.2% |
| tg128 | 20.63 | **36.01** | +74.4% |
| pp512 / pp1024 / pp2048 / pp4096 | 288.4 / 348.5 / 399.9 / 392.2 | **514.8 / 543.9 / 550.8 / 537.9** | +78.5 / +56.1 / +37.7 / +37.2% |
| Q8_0 of the same model, MTP mean over the same 4 prompts | 37.16 | **51.61** | +38.9% |
| Q8_0, per decode cycle | 11.11 | **15.51** | +39.7% |

What the shape of the table says:

- **Decode gains most.** +74.4% at tg128 and +48.2% per decode cycle under MTP, which is what a set built
  around GP100's missing `DP4A` and unused `HFMA2` should do.
- **Prefill gain narrows as the prompt grows** - +78.5% at pp512 down to +37.2% at pp4096 - because the
  set's advantage is at small batches and upstream's own prefill closes the distance at large ones. Take
  the pp row as four numbers, not one.
- **Not one quant ladder's luck.** At Q8_0, where nothing sits at 4 or 6 bits, MTP still gains +38.9% and
  +39.7% per cycle. Stock Q8_0 is in fact *faster* than stock Q6_K_XL on these prompts (37.16 against
  36.29), so the baseline is not a handicapped one - and the set's gain is smaller at Q8_0 than at
  Q6_K_XL, which is what you would expect when there are no 4-bit rows left for the HFMA2 paths to
  rescue.
- **The spread across prompts is the honest part**, and so is its size: +41.2% to +46.0% here. A single
  prompt is not a result, ours included - and the mean *is* sensitive to which four you committed, which
  is why these four now live in the repo (see the next bullet).
- **Committing the prompt set moved the headline, and the port did not.** The `b12211` table said +56.3%
  on four prompts that were never written down. Put both patched arms in one block, same prompts, same
  cards, same afternoon: `b12211`-patched 52.02 t/s against `v1.0.0`-patched 52.29, **+0.5%** - the port
  changed nothing. The whole 12-point difference in the headline is the prompt set: the same
  `b12211`-patched build that measured 56.45 on the old four measures 52.02 on these, 7.8% lower, while
  stock moves 0.5%. Those four prompts happened to sit where this build's decode path is strongest.
  +44.0% is what the committed harness reproduces, and the harness is now in the repo with its prompts.

One path is not carried over: the rollback/MTP conv-state store became a single `SET_ROWS` at `b12211`
(#31660) and the GDN half's snapshot branch still expects `K` separate copies, so `--spec-type draft-mtp`
refuses those layers on the 35B. That refusal was established on the `b12211` and `v0.8.0` arms and the code
that produces it crossed into `v1.0.0` unchanged; the three-card rows for that model are in
[docs/results.md](docs/results.md).

## Quality

PPL on WikiText-2 test, 16,384 tokens (`-c 2048 --chunks 8`), and logit KL against stock from the same
harness - the whole function's output at every position, which is what a greedy argmax hides. Measured on
the set **as a package** at `v1.0.0`, against stock `v1.0.0`, because that is what gets installed;
`benchmarks/kld-gate.sh` is the harness and [CONTRIBUTING.md](CONTRIBUTING.md) makes it a merge requirement
for every new patch or adopted candidate. The scale that makes a KL readable is a **quantization step on the
same corpus** - and at this ref it had to be re-taken under `-sm layer`, because stock `v1.0.0` refuses
`Q8_0` in a tensor split outright (`llama_params_fit is not implemented for SPLIT_MODE_TENSOR`, then
`GGML_ASSERT(meta_buf_ctx->bufs[i])` in `ggml-backend-meta.cpp`):

| quant | PPL stock | PPL this set | mean KLD vs stock | mean KLD of a **quantization step** against `Q8_0` |
|---|---:|---:|---:|---:|
| UD-Q4_K_M | 5.0101 | **4.9961** | 0.0061 +- 0.0004 | 0.0281 +- 0.0038 |
| UD-Q6_K_XL | 5.0881 | **5.0838** | 0.0102 +- 0.0016 | 0.0104 +- 0.0023 |

That last column is the answer to "how big is big": the quantization step you already accepted moves the
distribution **4.6x more** than the whole patch set does at Q4_K_M, and about the same as it does at
Q6_K_XL. The scale arm and the set-vs-stock arm are one split apart, which is the honest caveat - KL between
two quantizations of the same weights is a property of the weights, so the comparison stands, but it is not
one table measured one way. `UD-IQ2_XXS` is not re-listed here because it was not re-run at this ref; its
check is the `b12211` session's.

At the narrow batch shapes the decode kernels actually serve - 2 chunks, so a different token window and not
comparable to the prefill PPL above, but the same window on both arms - PPL moves by **+0.12% (Q4) and
+0.20% (Q6)** at the verify width and **-0.02% / -0.09%** at single-token decode, with mean KLD 4e-4 against
a harness floor that reads exactly zero on all four control arms (two of them as `-0.000000`, which is what
the gate's matcher used to swallow). Reproducibility, decided on the logits rather than the text:
`benchmarks/det-logits.sh` PASSes every verdict on both arms - the verify shape identical across three runs
and identical with graphs off, the prefill shape identical twice and again with graphs off. The two builds'
digests differ from each other (`0bcc38769019d57d` against `458c51df527c6734`) and that is the documented
`MEGA` arithmetic change, not a race: what matters is that each one repeats itself.

**Strict-output probe** - the failure mode PPL cannot see, 16 fixed prompts at temperature 0 (JSON that
must parse, exact tool and git commands, code completions, format constraints), re-run at this ref on
`v1.0.0` stock against `v1.0.0` this set: **11/16 on stock, 11/16 on this set, all 16 replies
byte-identical between the two builds**, no case passing on one and failing on the other, zero CUDA errors on
either. The absolute rate is not a capability claim - the validators assume particular key names, so a
correct-but-differently-spelled answer counts as a miss - but the comparison is what the probe is for, and
there is no difference to report.

So: a measurable distribution shift under a batched prefill, smaller than one quantization step, and not one
greedy token changed. That is what "the KL is a difference, not a degradation" looks like when you check it
two ways.

## Use it

```sh
./scripts/patch-llama.sh ~/llama.cpp-patched --build      # clone v1.0.0, apply 79 patches, build for sm_60
```

The script checks out the tag this set is maintained against, refuses to touch the tree if the commit does
not match, strips CRLF off the patch files if you cloned on Windows, applies with `patch -p1 -F0` and stops
at the first failure. Add `--check` to test the apply without leaving anything behind, and `--ref` /
`--commit` if you are deliberately aiming at another tree. By hand it is three commands:

```sh
git clone --branch v1.0.0 --depth 1 https://github.com/ggml-org/llama.cpp && cd llama.cpp
for p in /path/to/this/repo/patches/*.patch; do patch -p1 -F0 < "$p" || break; done
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 -DGGML_NATIVE=ON -DGGML_NCCL=OFF -DGGML_CUDA_NCCL=OFF
```

Two flags are not decoration. CUDA **12.x**: CUDA 13 dropped Pascal code generation, and a build whose
`cudaCapabilities` start at 7.5 leaves no `sm_60` code in the binary and fails at runtime with *"named
symbol not found"*. And `GGML_CUDA_NCCL=OFF`: with NCCL installed, leaving it at its default makes every
`-sm tensor` run die in `graph_compute` with `unhandled cuda error`, which reads exactly like a regression
in the patch set and is not ([why](docs/benchmarking.md#an-arm-can-silently-become-another-arm)).

A patch set applied to a different llama.cpp ref is not this set. Expect rejects rather than a partial
success, and keep `-F0` - a fuzzy hunk can land in the wrong place and still exit 0.

## Switches

Everything above is the default; no environment variable is needed to get these numbers. Every patch is on
by default with a runtime way back - `=0` to disable, never a build-time option - and
[docs/patches.md](docs/patches.md) lists the sub-switches per patch.

| variable | default | off means |
|---|---|---|
| `GGML_CUDA_MMVQ_P100_BODY=0` | on | patch 90's adopted MMVQ kernel body, back to llama.cpp's own `mul_mat_vec_q` |
| `GGML_CUDA_MMVQ_F16=0` | on | patch 91's fp16 verify matvec path (Q6_K and the generic types) |
| `GGML_CUDA_P60_MMVQ=0` | on | the fp16 decode matvec path and everything fused into it |
| `GGML_CUDA_P60_MEGA=0` | on | the persistent layer-half kernels and the prefill GEMM/GDN/FA kernels |
| `GGML_CUDA_P60_MEGAT=0` | on | the persistent kernels taking 2..6-token batches (the MTP verify step) |
| `GGML_CUDA_P60_GDN_HALF=0` | on | the persistent gated-delta-net half (+22.9% MoE decode) |
| `GGML_CUDA_AR_PASCAL=0` | on | patch 49's Pascal AllReduce gate, back to upstream's reduction |
| `GGML_CUDA_AR_H2D_STREAM=0` | on | the copy-engine return leg on a second stream |
| `GGML_CUDA_CUBLAS_ALGO_PASCAL=-1` | on | patch 51's cuBLAS algorithm choice at prefill sizes |
| `GGML_CUDA_FUSE_GDN_GATHER=1` | off | patch 24's deferred GDN state gather, on for measurement only - see below |
| `GGML_CUDA_P60_EXACT=1` | off | the bit-exact mode: decode and verify identical to stock llama.cpp, slow by design |
| `GGML_CUDA_DISABLE_GRAPHS=1` | off | all CUDA graph capture (the way to A/B anything that touches a graph) |

Patch 92 rides on `GGML_CUDA_MMVQ_P100_BODY` - it has no variable of its own, and `=0` returns Q6_K to the
vdr-2 dot. Patch 22 self-disables under `-sm tensor` (the meta device does not tolerate more than one graph
per compute-buffer rotation), so the tensor rows above are the set minus 22; `LLAMA_DEC_SLOTS=0` is the
manual way back and `LLAMA_DEC_MAX_TOK` the width knob. Patch 12 - the HFMA2 Q4_1 GEMV - is reached only
with `GGML_CUDA_MMVQ_P100_BODY=0`, because 90 takes the whole function before 12's hook runs; that is
measured, documented, and the one place in this set where the default routes around a ported path
([docs/p60-open-items.md](docs/p60-open-items.md) item 5).

## Known limits

Full text, with the measurements behind each, in [docs/results.md](docs/results.md) and
[docs/benchmarking.md](docs/benchmarking.md).

- **Decode-tuned.** Single-stream decode with speculative verify batches of 2-5. Large-batch serving was
  never the target.
- **The P60 block is one model family.** Qwen3.5/3.6 MoE at 2048 hidden with Q4_K/Q5_K experts, `sm_60`
  gated as a whole; its matchers key on weight names and shapes and refuse anything else - which is
  measured, not assumed: the dense 27B is flat between the two builds.
- **`MEGA` changes the answer.** The persistent halves are a different arithmetic path - patch 87's GDN
  half is mean KLD 0.0284 against the per-kernel path. Reproducible, identical with and without graph
  replay, and not llama.cpp's numbers. `GGML_CUDA_P60_EXACT=1` is the way back.
- **Patch 24 ships off.** With the deferred GDN state gather on, the same binary, flags and corpus gave
  three different logits digests in 2-6 token batches. Switching it off costs nothing measurable.
- **qwen4exp prefill loses.** Measured once on a 116 B A3B MoE at `-p 512`: -5.1% tensor / -6.1% layer
  prefill while buying +13.1% / +20.3% decode. The responsible patch is not identified.
- **Pascal only.** Patches scoped `sm_60`, `pre-Volta` or `pre-Turing` change values upstream tuned for
  newer hardware, and `pre-Turing` includes Volta, never measured. Patch 07 is not gated at all.
- **One box.** 2 and 3 P100s at 160 W. Nothing has run on four cards.
- **The correctness suite says nothing changed.** `test-backend-ops` at `v1.0.0`, op by op on both arms,
  whole-op runs plus a per-type split of `MUL_MAT` and `MUL_MAT_ID` over 16 weight types: every case that
  runs runs identically on both builds (`GET_ROWS` 227/227, `ROPE` 490/490, `MUL_MAT_ID` 1040/1040, and no
  type where the counts differ). The suite still cannot be driven as one process: one upstream case aborts
  on `GGML_ASSERT(convert_func != nullptr)` in `ggml-cuda.cu` on stock and patched alike, and with worker
  threads that abort truncates each arm's log at a different place - so a whole-`MUL_MAT` comparison
  proves nothing and the per-type runs are what the claim rests on.

## Where things live

```
patches/            79 .patch files, applied in numeric order (52-89 are the P60 block, 90-92 the adoption)
scripts/patch-llama.sh   clone llama.cpp at the pinned ref, apply the set, optionally build
docs/patches.md     per-patch index: scope, effect, kill switch, what it changes, dead ends
docs/results.md     every session: trees, flags, power, raw rows - and the caveats in full
docs/benchmarking.md  the measurement discipline this set needed: thermal control, noise floors,
                      how an arm silently becomes another arm, why two predictions were wrong
docs/p60-open-items.md  what the P60 block still needs, with the measurements behind it
docs/p60-series.tsv 52-83 mapped to the upstream commits they came from
benchmarks/         how to re-measure (run.sh for throughput, mtp-ab.sh + mtp-prompts.txt for the
                    speculative table, ppl.sh, kld-gate.sh for the package PPL+KL quality gate,
                    knob-sweep.sh for the environment switches, agentic-probe.py for strict outputs,
                    logit-identity.sh and det-logits.sh for determinism) and results/ for each release
RELEASING.md        what to do when llama.cpp ships a new version
CONTRIBUTING.md     adding a patch, reporting a measurement, what a good PR looks like
```

## Status of the release

All 79 are verified by clean-room apply: a fresh clone at `v1.0.0` plus these files at `-F0` gives 0
rejects and 0 offset hunks, and the resulting tree is byte-identical to the one reviewed patch by patch,
which is what `./scripts/patch-llama.sh --check` runs - it now fails on an offset, not only on a reject,
because an accepted offset means the file moved under the hunk and the next rebase can no longer tell
"the lines shifted" from "the code changed". Of the 79, 18 applied byte-identical to the previous ref, 52
were re-cut at exact context and 9 needed real authoring: **10** and **21** because upstream added
parameters and a third per-tensor table to what they patch, **22** because `sched_reserve()` was rewritten
around a `sched_params` struct, **35**, **36**, **54** and **77** because the call sites they hook moved or
grew guards, and **86** and **90** because `ggml_cuda_graph` moved out of `common.cuh` into `graph.cuh` and
the cuda-context constructor moved into the `.cu`. Two of those were only visible at the build: the set
calls `quantize_row_q8_1_cuda`, which v1.0.0 stripped two parameters from, and `sched_new`, which it turned
into a struct - both applied at zero fuzz and zero offset and then failed to compile, which is the whole
argument for building as a gate. `benchmarks/knob-sweep.sh` reports 75 of the 76 environment knobs the set
introduces still read by the code; the exception, `GGML_CUDA_P60_MEGA_GRAPHS`, is introduced by 85 and
deleted by 86 in the same set. Both arms - pristine `v1.0.0` and the applied set - are **built** at `sm_60`
under CUDA 12.4 with NCCL off, and all eight non-CUDA translation units the set touches pass
`g++ -fsyntax-only`. Patch 24 stays opt-in for the reproducibility reason above. The `b12211` tag
is the previous release - 79 files at `7d7f028`, on the `b12211` branch as well as its tag - and `main` now
carries the next ref, not a revision of it.
