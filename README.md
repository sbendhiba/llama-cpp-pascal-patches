# llama-cpp-pascal-patches

Performance patches for [llama.cpp](https://github.com/ggml-org/llama.cpp), developed and measured on
Tesla P100 (GP100, `sm_60`). Not a fork: one `.patch` file per change, applied in numeric order with zero
fuzz, each carrying its own measurement, its kill switch and its dead ends in the comments it adds to the
source. Unofficial, unaffiliated with the llama.cpp project, MIT.

GP100 is the reason any of this exists. Alone among Pascal chips it has no `DP4A` and has full-rate
`HFMA2`, so llama.cpp's quantized matmul paths fall back to int emulation and cuBLAS while the one
instruction the card is genuinely good at goes unused. The rest is what the last two years of llama.cpp
made possible: a fused norm, deferred gathers, CUDA graphs, speculative decoding with an MTP head.

This branch is the **`b12211`** set - **79 patches** against llama.cpp `7d7f028` (b12211, 101 commits past
`v0.9.0`): 39 we carry, the 37-patch **P60 block** (52-89, a port and retune of
[Mrinohk/qwen36-p100-patches](https://github.com/Mrinohk/qwen36-p100-patches) and
[qr0n/p100-llama-cpp](https://github.com/qr0n/p100-llama-cpp) work for the same cards) and **90-92**,
adopted from [junkgpulab/llama.cpp-p100](https://github.com/junkgpulab/llama.cpp-p100). Two patches went
at this ref because upstream made them unnecessary: **59** (the MoE expert-tile rule is now unconditional
in `mmq.cu`) and **32** (upstream added its own split-input prefetch, same field names). Other tags are
other refs: `git tag -l` gives `v0.6.0`, `v0.7.0`, `v0.8.0`, `v0.9.0` and the prerelease line `b11601`.

## Speed

3x Tesla P100-PCIE-16GB at **160 W**, CUDA 12.4, `-fa on -t 28 -ub 2048`, llama-bench, two cards unless
stated, all arms in one session so a release delta and a port delta are not read out of two different
afternoons ([docs/results.md](docs/results.md)).

| model | split | stock `b12211` pp512 / tg128 | this set pp512 / tg128 |
|---|---|---:|---:|
| Cyber-Tiel-Coder-35B-A3B MTP UD-Q4_K_XL (MoE) | `-sm layer` | 658.04 / 58.19 | **1060.92 / 119.22** (+61% / +105%) |
| | `-sm tensor` | 989.23 / 58.86 | **1565.70 / 84.65** (+58% / +44%) |
| Qwen3.8-27B UD-Q4_K_M (dense) | `-sm tensor` | 230.15 / 20.95 | 378.16 / 29.61 (+64% / +41%) |
| | `-sm layer` | 144.30 / 13.22 | 207.41 / 18.32 (+44% / +39%) |

**The MoE prefill percentage fell because upstream got faster, not because this set got slower.** Stock
`b12211` prefill is 658 / 989 against `v0.9.0`'s 489 / 766; this set's own prefill is unchanged within
noise, so the ratio drops from +119% to +61% at the same absolute throughput. Decode is flat across the
ref on all four configs, and against its own `v0.9.0` build this set is within 1.1% on every cell -
and `benchmarks/det-logits.sh` returns the *same* logits digests as `v0.9.0` on both batch shapes, four
PASS, with the persistent halves actually running.

### Qwen3.8-27B with the MTP head

`llama-server` `/completion` at temperature 0, `--spec-type draft-mtp --spec-draft-n-max 4
--spec-draft-p-min 0.2`, 3-card tensor, 4 prompts x 256 tokens x 2 reps x **both arm orders**, every arm
identity-checked so a request cannot be answered by the previous build's server. Per-cycle divides
throughput by the mean draft length, which takes speculative acceptance out of the number.

| UD-Q6_K_XL, 27B | stock `b12211` | **this set (79)** | gain |
|---|---:|---:|---:|
| MTP, mean over 4 prompts | 36.11 t/s | **56.45** | **+56.3%** |
| MTP per draft cycle | 10.36 | **16.23** | +56.7% |
| MTP, code prompt | 44.42 | **69.51** | +56.5% |
| MTP, prose / haiku / translation | 27.77 / 37.94 / 34.32 | **43.09 / 55.16 / 58.03** | +55.2 / +45.4 / +69.1% |
| tg128 | 20.44 | **35.64** | +74.4% |
| pp512 / pp1024 / pp2048 / pp4096 | 286.1 / 345.7 / 396.3 / 389.2 | **514.5 / 546.2 / 552.3 / 537.4** | +79.8 / +58.0 / +39.3 / +38.1% |
| Q8_0 of the same model, MTP code + chat | 36.06 | **54.67** | +51.6% |

What the shape of the table says:

- **Decode gains most.** +74.4% at tg128 and +56.3% under MTP, which is what a set built around GP100's
  missing `DP4A` and unused `HFMA2` should do.
- **Prefill gain narrows as the prompt grows** - +79.8% at pp512 down to +38.1% at pp4096 - because the
  set's advantage is at small batches and upstream's own prefill closes the distance at large ones. Take
  the pp row as four numbers, not one.
- **Not one quant ladder's luck.** At Q8_0, where nothing sits at 4 or 6 bits, MTP still gains +51.6%;
  stock Q8_0 and stock Q6_K_XL run within 0.2% of each other here, so both arms face the same hardware
  limit and the comparison is not against a handicapped baseline.
- **The spread across prompts is the honest part.** The per-prompt gains run +45.4% to +69.1%: prompt
  length and how well the MTP head drafts a given style both move the number, so a single-prompt figure
  of any kind - ours or anybody else's - is not a result.

One path is not carried over yet: the rollback/MTP conv-state store became a single `SET_ROWS` at this ref
(#31660) and the GDN half's snapshot branch still expects `K` separate copies, so `--spec-type draft-mtp`
refuses those layers on the 35B. Three-card rows are in [docs/results.md](docs/results.md).

## Quality

PPL on WikiText-2 test, 16,384 tokens (`-c 2048 --chunks 8`), and logit KL against stock from the same
harness - the whole function's output at every position, which is what a greedy argmax hides. Measured on
the set **as a package**, against stock, because that is what gets installed; `benchmarks/kld-gate.sh` is
the harness and [CONTRIBUTING.md](CONTRIBUTING.md) makes it a merge requirement for every new patch or
adopted candidate. The reference that makes these numbers readable is already in this file: **patch 87's
persistent GDN half alone is documented at mean KLD 0.0284** against the per-kernel path, and a
stock-vs-stock control of this harness returns 0.000000, so these are real differences, not measurement
noise.

| quant | PPL stock | PPL this set | mean KLD vs stock | mean KLD of a **quantization step** on the same harness |
|---|---:|---:|---:|---:|
| UD-Q4_K_M | 5.0606 | **5.0475** | 0.0066 | 0.0306 (vs `Q8_0`) |
| UD-Q6_K_XL | 5.0881 | **5.0838** | 0.0102 | 0.0103 (vs `Q8_0`) |
| UD-IQ2_XXS | 5.7274 | **5.6910** | better on all 8 chunks | - |

That column is the answer to "how big is big": the quantization step you already accepted moves the
distribution 4.6x more than the whole patch set does at Q4_K_M, and about the same at Q6_K_XL. Those two
files also differ in more than the quant, so treat the column as a reference scale rather than a controlled
ablation.

At the narrow batch shapes the decode kernels actually serve, PPL moves by 0.05% (Q4) and 0.20% (Q6)
and KLD by 0.0004, against a harness floor of exactly zero. For scale: choosing `-ub 8` over `-ub 512` moves
stock's *own* PPL by 1.3%, six times the largest difference between the builds.

**Strict-output probe** - the failure mode PPL cannot see, 16 fixed prompts at temperature 0 (JSON that
must parse, exact tool and git commands, code completions, format constraints): **11/16 passing on stock
and 11/16 on this set, with all 16 replies byte-identical between the two builds** and zero CUDA errors on
either. The absolute rate is not a capability claim - the validators assume particular key names, so a
correct-but-differently-spelled answer counts as a miss - but the comparison is what the probe is for, and
there is no difference to report: no case passes on one build and fails on the other.

So: a measurable distribution shift under a batched prefill, and not one greedy token changed. That is
what "the KL is a difference, not a degradation" looks like when you check it two ways.

## Use it

```sh
./scripts/patch-llama.sh ~/llama.cpp-patched --build      # clone b12211, apply 79 patches, build for sm_60
```

The script checks out the tag this set is maintained against, refuses to touch the tree if the commit does
not match, strips CRLF off the patch files if you cloned on Windows, applies with `patch -p1 -F0` and stops
at the first failure. Add `--check` to test the apply without leaving anything behind, and `--ref` /
`--commit` if you are deliberately aiming at another tree. By hand it is three commands:

```sh
git clone --branch b12211 --depth 1 https://github.com/ggml-org/llama.cpp && cd llama.cpp
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
vdr-2 dot. Patch 22 self-disables under `-sm tensor` (the meta backend does not tolerate more than one
graph per compute-buffer rotation), so the tensor rows above are the set minus 22.

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
- **The correctness suite says nothing changed.** `test-backend-ops` at `v0.9.0`, op by op on both arms:
  130 ops, 14,822 passing cases each, zero rows where the arms differ. The one crash is upstream's:
  `MUL_MAT(q4_K, m=8192, n=9)` aborts on `GGML_ASSERT(convert_func != nullptr)` on a Pascal card in stock
  `v0.9.0` and `v0.8.0`, which is why the suite must be driven op by op.

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
benchmarks/         how to re-measure (run.sh for throughput, ppl.sh, kld-gate.sh for the package
                    PPL+KL quality gate, agentic-probe.py for strict outputs, logit-identity.sh and
                    det-logits.sh for determinism) and results/ for each release
RELEASING.md        what to do when llama.cpp ships a new version
CONTRIBUTING.md     adding a patch, reporting a measurement, what a good PR looks like
```

## Status of this branch

All 79 are verified by clean-room apply: a fresh clone plus these files at `-F0` gives 0 rejects and a
tree byte-identical to the one reviewed patch by patch, which is what `./scripts/patch-llama.sh --check`
runs. 382 of 383 identifiers resolve (the exception, `GGML_CUDA_P60_MEGA_GRAPHS`, is introduced by 85 and
deleted by 86 in the same set), the host-side syntax check is clean on all eight non-CUDA translation
units, and both stock and patched trees are built at `sm_60` under CUDA 12.4 with NCCL off. Five patches
needed real authoring at this ref - 33, 35, 42, 71 and 83 - and 31 taught the build gate its job: it
applied at zero fuzz and zero offset and then failed to compile, because the kernel it launches had grown a
parameter. Patch 24 stays opt-in for the reproducibility reason above. The `v0.9.0` tag is that release -
78 files, patch 24 off - and this branch is the next ref, not a revision of it.
