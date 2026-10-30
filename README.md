# llama-cpp-pascal-patches

Performance patches for [llama.cpp](https://github.com/ggml-org/llama.cpp), developed and measured on
Tesla P100 (GP100, `sm_60`). Not a fork: one `.patch` file per change, applied in numeric order with
zero fuzz, each carrying its own measurement, its kill switch and its dead ends in the comments it adds
to the source. Unofficial, unaffiliated with the llama.cpp project, MIT.

GP100 is the reason any of this exists. Alone among Pascal chips it has no `DP4A` and has full-rate
`HFMA2`, so llama.cpp's quantized matmul paths fall back to int emulation and cuBLAS while the one
instruction the card is genuinely good at goes unused. The rest is what the last two years of llama.cpp
made possible: a fused norm, deferred gathers, CUDA graphs, speculative decoding with an MTP head.

This branch is the **`v0.8.0`** set - **78 patches** against llama.cpp `8642541b2`: 40 we carry (of which
10 are Pascal-gated, 29 are hardware-neutral, one changes a constant every GPU sees) plus the 38-patch
**P60 block** (52-89), a port and retune of [Mrinohk/qwen36-p100-patches](https://github.com/Mrinohk/qwen36-p100-patches)
and [qr0n/p100-llama-cpp](https://github.com/qr0n/p100-llama-cpp) work for the same cards. Other branches
and tags are other refs: `git tag -l` lists `v0.6.0`, `v0.7.0`, `v0.8.0` and the prerelease line `b11601`.

## What you get

3x Tesla P100-PCIE-16GB at **160 W**, CUDA 12.4, `-fa on -t 28 -ub 2048`, llama-bench, two cards unless
stated. `stock` rows and the `v0.8.0`-era rows of this set come from the three-arm session of 2026-10-30
([docs/results.md](docs/results.md)); the last column is the 78-patch build measured the same day against
the 74-patch one, so a 4-patch delta cannot hide inside a session change:

| model | split | stock pp512 / tg128 | this set pp512 / tg128 |
|---|---|---:|---:|
| Cyber-Tiel-Coder-35B-A3B MTP UD-Q4_K_XL (MoE) | `-sm layer` | 488.75 / 58.34 | **1075.38 / 119.89** (+120% / +105%) |
| | `-sm tensor` | 767.96 / 59.78 | **1600.59 / 87.13** (+108% / +46%) |
| | `-sm layer`, 3 cards | - | 1077.12 / 118.83 |
| Qwen3.8-27B UD-Q4_K_M (dense) | `-sm tensor` | 230.10 / 20.99 | 382.82 / 30.01 (+66% / +43%) |
| | `-sm layer` | 143.80 / 13.21 | 208.82 / 18.52 (+45% / +40%) |

Speculative decode with the model's own MTP head (`--spec-type draft-mtp`, 4 slots): **99.43 t/s**
against 79.04 with the last two switches turned off and the same draft acceptance. Every number above is
a mean over interleaved rounds with the cards cooled between runs; sessions, flags and raw rows are in
[docs/results.md](docs/results.md).

## Use it

```sh
./scripts/patch-llama.sh ~/llama.cpp-patched --build      # clone v0.8.0, apply 78 patches, build for sm_60
```

The script checks out the tag this set is maintained against, refuses to touch the tree if the commit
does not match, strips CRLF off the patch files if you cloned on Windows, applies with `patch -p1 -F0`
and stops at the first failure. Add `--check` to test the apply without leaving anything behind, and
`--ref` / `--commit` if you are deliberately aiming at another tree. By hand it is three commands:

```sh
git clone --branch v0.8.0 --depth 1 https://github.com/ggml-org/llama.cpp && cd llama.cpp
for p in /path/to/this/repo/patches/*.patch; do patch -p1 -F0 < "$p" || break; done
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 -DGGML_NATIVE=ON -DGGML_NCCL=OFF -DGGML_CUDA_NCCL=OFF
```

Two things in that flag list are not decoration. CUDA **12.x**: CUDA 13 dropped Pascal code generation,
and a build whose `cudaCapabilities` start at 7.5 leaves no `sm_60` code in the binary and fails at
runtime with *"named symbol not found"*. And `GGML_CUDA_NCCL=OFF`: on a box that has NCCL installed,
leaving it at its default makes every `-sm tensor` run die in `graph_compute` with `unhandled cuda
error`, which reads exactly like a regression in the patch set and is not
([why](docs/benchmarking.md#an-arm-can-silently-become-another-arm)).

A patch set applied to a different llama.cpp ref is not this set. The patches are generated at exact
context; expect rejects rather than a partial success, and keep `-F0` - fuzzy application can land a
hunk in the wrong place and still exit 0.

## Switches

Everything measured here is the default: no environment variable is needed to get the numbers above.
Each one has a way back, and [docs/patches.md](docs/patches.md) lists the sub-switches per patch.

| variable | default | off means |
|---|---|---|
| `GGML_CUDA_P60_MMVQ=0` | on | the fp16 decode matvec path and everything fused into it |
| `GGML_CUDA_P60_MEGA=0` | on | the persistent layer-half kernels and the prefill GEMM/GDN/FA kernels |
| `GGML_CUDA_P60_MEGAT=0` | on | the persistent kernels taking 2..6-token batches (the MTP verify step) |
| `GGML_CUDA_P60_GDN_HALF=0` | on | the persistent gated-delta-net half (+22.9% MoE decode) |
| `GGML_CUDA_AR_PASCAL=0` | on | patch 49's Pascal AllReduce gate, back to upstream's reduction |
| `GGML_CUDA_AR_H2D_STREAM=0` | on | the copy-engine return leg on a second stream |
| `GGML_CUDA_CUBLAS_ALGO_PASCAL=-1` | on | patch 51's cuBLAS algorithm choice at prefill sizes |
| `GGML_CUDA_P60_EXACT=1` | off | their bit-exact mode: decode and verify identical to stock llama.cpp, slow by design |
| `GGML_CUDA_DISABLE_GRAPHS=1` | off | all CUDA graph capture (the way to A/B anything that touches a graph) |

## Read this before trusting the numbers

- **Decode-tuned.** Single-stream decode, with speculative verify batches of 2-5. Large-batch serving
  was never the target.
- **The P60 block is one model family.** Qwen3.5/3.6 MoE at 2048 hidden with Q4_K/Q5_K experts, `sm_60`
  gated as a whole. Its matchers key on weight names and shapes and refuse anything else, so other
  models take the normal path - which is measured, not assumed: the dense 27B is flat between the two
  builds.
- **`MEGA` changes the answer.** The persistent halves are a different arithmetic path. Turning the
  block on changes greedy output (mean KLD 0.0284 per token for the GDN half against the per-kernel
  path); it is reproducible run to run and identical with and without graph replay, but it is not
  llama.cpp's numbers. `GGML_CUDA_P60_EXACT=1` is the bit-exact mode.
- **qwen4exp prefill loses.** Measured once, on a 116 B A3B MoE at `-p 512`: the set costs -5.1% (tensor)
  / -6.1% (layer) there while buying +13.1% / +20.3% decode. The responsible patch is not identified, so
  if prefill on that architecture matters, measure before you deploy.
- **Pascal only.** Patches scoped `sm_60`, `pre-Volta` or `pre-Turing` change values upstream tuned for
  newer hardware, and `pre-Turing` includes Volta, which was never measured. Patch 07 is not gated at
  all. Measure before using any of them on Turing and later.
- **One box.** 2 and 3 P100s at 160 W. Nothing has been run on four cards, and patch 49's N-way reduce
  was validated by its author on virtual devices before it ran on three real ones.
- **No correctness suite has been run.** `test-backend-ops` has never been pointed at these trees, and
  no perplexity number sits beside most of these gains. Verification here is throughput, output hashing,
  logit KLD and reading the graph.
- One patch self-limits: 22 disables itself under `-sm tensor` (it conflicts with the meta backend), so
  the `-sm tensor` rows above are the set minus 22. Single-card rows include it.

## Where things live

```
patches/            78 .patch files, applied in numeric order (52-89 are the P60 block)
scripts/patch-llama.sh   clone llama.cpp at the pinned ref, apply the set, optionally build
docs/patches.md     per-patch index: scope, effect, kill switch, what it changes, dead ends
docs/results.md     every session: trees, flags, power, raw rows - and the caveats in full
docs/benchmarking.md  the measurement discipline this set needed: thermal control, noise floors,
                      how an arm silently becomes another arm, why two predictions were wrong
docs/p60-open-items.md  what the P60 block still needs, with the measurements behind it
docs/p60-series.tsv 52-83 mapped to the upstream commits they came from
benchmarks/         how to re-measure (run.sh, ppl.sh, logit-identity.sh, the determinism and
                    gather-contract tests) and results/ for each release
RELEASING.md        what to do when llama.cpp ships a new version
CONTRIBUTING.md     adding a patch, reporting a measurement, what a good PR looks like
```

## Status of this branch

All 40 of ours and the 38-patch P60 block are regenerated against llama.cpp `v0.8.0` (`8642541b2`) and
verified by clean-room apply: a fresh clone plus these 78 files at `-F0` gives 0 rejects, 52 files
changed, and a tree byte-identical to the one every number above was measured on - which is what
`./scripts/patch-llama.sh --check` runs. Both stock and patched trees have been built at `sm_60` under
CUDA 12.4 and run on the cards.
