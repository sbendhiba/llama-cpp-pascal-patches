# llama-cpp-pascal-patches

Performance patches for [llama.cpp](https://github.com/ggml-org/llama.cpp), developed and measured on
Tesla P100 (GP100, `sm_60`). Not a fork: one `.patch` file per change, applied in numeric order with
zero fuzz, each carrying its own measurement, its kill switch and its dead ends in the comments it adds
to the source. Unofficial, unaffiliated with the llama.cpp project, MIT.

GP100 is the reason any of this exists. Alone among Pascal chips it has no `DP4A` and has full-rate
`HFMA2`, so llama.cpp's quantized matmul paths fall back to int emulation and cuBLAS while the one
instruction the card is genuinely good at goes unused. The rest is what the last two years of llama.cpp
made possible: a fused norm, deferred gathers, CUDA graphs, speculative decoding with an MTP head.

This branch is the **`b12211`** set - **76 patches** against llama.cpp `7d7f028` (b12211, 101 commits
past `v0.9.0`): 39 we carry (of which 10 are Pascal-gated, 28 are hardware-neutral, one changes a
constant every GPU sees) plus the 37-patch **P60 block** (52-89), a port and retune of
[Mrinohk/qwen36-p100-patches](https://github.com/Mrinohk/qwen36-p100-patches) and
[qr0n/p100-llama-cpp](https://github.com/qr0n/p100-llama-cpp) work for the same cards. Two patches went
at this ref because upstream made them unnecessary: **59** (the MoE expert-tile rule is now
unconditional in `mmq.cu`) and **32** (upstream added its own split-input prefetch, with the same field
names). Other branches and tags are other refs: `git tag -l` lists `v0.6.0`, `v0.7.0`, `v0.8.0`,
`v0.9.0` and the prerelease line `b11601`; this branch is `b12211`.

## What you get

3x Tesla P100-PCIE-16GB at **160 W**, CUDA 12.4, `-fa on -t 28 -ub 2048`, llama-bench, two cards unless
stated. Measured 2026-11-08 with three arms in one session - stock `b12211`, this set at `b12211`, and
this set at `v0.9.0` beside them, so a release delta and a port delta are not read out of two sessions
([docs/results.md](docs/results.md)).

| model | split | stock b12211 pp512 / tg128 | this set pp512 / tg128 |
|---|---|---:|---:|
| Cyber-Tiel-Coder-35B-A3B MTP UD-Q4_K_XL (MoE) | `-sm layer` | 658.04 / 58.19 | **1060.92 / 119.22** (+61% / +105%) |
| | `-sm tensor` | 989.23 / 58.86 | **1565.70 / 84.65** (+58% / +44%) |
| Qwen3.8-27B UD-Q4_K_M (dense) | `-sm tensor` | 230.15 / 20.95 | 378.16 / 29.61 (+64% / +41%) |
| | `-sm layer` | 144.30 / 13.22 | 207.41 / 18.32 (+44% / +39%) |

**The MoE prefill percentage fell because upstream got faster, not because this set got slower.**
Stock `b12211` prefill is 658 / 989 against `v0.9.0`'s 489 / 766 (+35% / +29%, from #31047's
linearized MoE layout and the expert-tile rule that used to be patch 59); this set's own prefill is
unchanged within noise, so the ratio drops from +119% to +61% at the same absolute throughput.
Decode is flat across the ref on all four configs. Against its own `v0.9.0` build this set is within
1.1% on every cell, and `benchmarks/det-logits.sh` gives the *same* logits digests as `v0.9.0` on both
batch shapes - four PASS, with the persistent halves actually running.

One path is not carried over yet: the rollback/MTP conv-state store became a single `SET_ROWS` at
this ref (#31660), and the GDN half's snapshot branch still expects `K` separate copies, so
`--spec-type draft-mtp` refuses those layers. The `v0.9.0` MTP numbers do not apply here and have not
been re-measured. Three-card rows are in [docs/results.md](docs/results.md).

## Use it

```sh
./scripts/patch-llama.sh ~/llama.cpp-patched --build      # clone b12211, apply 76 patches, build for sm_60
```

The script checks out the tag this set is maintained against, refuses to touch the tree if the commit
does not match, strips CRLF off the patch files if you cloned on Windows, applies with `patch -p1 -F0`
and stops at the first failure. Add `--check` to test the apply without leaving anything behind, and
`--ref` / `--commit` if you are deliberately aiming at another tree. By hand it is three commands:

```sh
git clone --branch b12211 --depth 1 https://github.com/ggml-org/llama.cpp && cd llama.cpp
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
| `GGML_CUDA_FUSE_GDN_GATHER=1` | off | patch 24's deferred GDN state gather, back on for measurement only - see the reproducibility note below |
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
- **Patch 24 ships switched off.** `fuse-gdn-state-gather` defers a node to its consumer, ten places
  down the graph, and with it on the *same* binary, flags and corpus gave three different logits
  digests in 2-6 token batches - PPL 8.70 to 8.87 between runs of the same thing. Switching it off
  costs nothing measurable: identical perplexity on WikiText-2 (6.5905 on, 6.5905 off at `-sm layer`;
  6.5894 both at `-sm tensor`), and at most 1% of decode (MoE layer -0.2%, MoE tensor -1.9%, dense
  -1.0%), prefill inside the noise. So the default is off and `GGML_CUDA_FUSE_GDN_GATHER=1` puts it
  back for measurement. Re-checked after the flip with `benchmarks/det-logits.sh`: three identical
  runs of the shipped build hash the same, at `v0.9.0` and at `v0.8.0`, and the fold switched back on
  diverges again - which is the part that makes this a patch problem and not a card problem.

- **qwen4exp prefill loses.** Measured once, on a 116 B A3B MoE at `-p 512`: the set costs -5.1% (tensor)
  / -6.1% (layer) there while buying +13.1% / +20.3% decode. The responsible patch is not identified, so
  if prefill on that architecture matters, measure before you deploy.
- **Pascal only.** Patches scoped `sm_60`, `pre-Volta` or `pre-Turing` change values upstream tuned for
  newer hardware, and `pre-Turing` includes Volta, which was never measured. Patch 07 is not gated at
  all. Measure before using any of them on Turing and later.
- **One box.** 2 and 3 P100s at 160 W. Nothing has been run on four cards, and patch 49's N-way reduce
  was validated by its author on virtual devices before it ran on three real ones.
- **The correctness suite is now run, and it says nothing changed.** `test-backend-ops` at `v0.9.0`, one
  op at a time on both arms: 130 ops, **14,822 passing cases on each**, zero rows where the arms differ
  (`FLASH_ATTN_EXT` 4041/4041, `MUL_MAT_ID` 932/932, `SET_ROWS` 160/160, `GATED_DELTA_NET` 44/44). The one
  crash is upstream's, not ours: `MUL_MAT(type_a=q4_K, m=8192, n=9, k=512, per=[0,1,2,3])` aborts on
  `GGML_ASSERT(convert_func != nullptr)` on a Pascal card in stock `v0.9.0`, and did so in stock `v0.8.0`
  at its own line 1516 - which is why the suite has to be driven op by op, or that one case ends the run.
  No perplexity number sits beside most of these gains; verification is still throughput, output hashing,
  logit KLD and reading the graph.
- One patch self-limits: 22 disables itself under `-sm tensor` (it conflicts with the meta backend), so
  the `-sm tensor` rows above are the set minus 22. Single-card rows include it.

## Where things live

```
patches/            76 .patch files, applied in numeric order (52-89 are the P60 block)
scripts/patch-llama.sh   clone llama.cpp at the pinned ref, apply the set, optionally build
docs/patches.md     per-patch index: scope, effect, kill switch, what it changes, dead ends
docs/results.md     every session: trees, flags, power, raw rows - and the caveats in full
docs/benchmarking.md  the measurement discipline this set needed: thermal control, noise floors,
                      how an arm silently becomes another arm, why two predictions were wrong
docs/p60-open-items.md  what the P60 block still needs, with the measurements behind it
docs/p60-series.tsv 52-83 mapped to the upstream commits they came from
benchmarks/         how to re-measure (run.sh, ppl.sh, logit-identity.sh, det-logits.sh - the logits
                    determinism probe - and the determinism and gather-contract tests) and results/ for
                    each release
RELEASING.md        what to do when llama.cpp ships a new version
CONTRIBUTING.md     adding a patch, reporting a measurement, what a good PR looks like
```

## Status of this branch

All 76 are regenerated against llama.cpp `b12211` (`7d7f028`) and verified by clean-room apply: a
fresh clone plus these files at `-F0` gives 0 rejects and a tree byte-identical to the one reviewed
patch by patch - which is what `./scripts/patch-llama.sh --check` runs. The host-side syntax check is
clean on all eight non-CUDA translation units, 382 of 383 identifiers resolve (the one that does not is
introduced by 85 and deleted by 86 in the same set), and both stock and patched trees are built at
`sm_60` under CUDA 12.4 with NCCL off.

Nothing was deleted at this ref: every patch either moved or stayed where it was. Four needed real
authoring rather than a re-cut - **31** (the f16 KV chunk, against upstream's new `kv_rows` multi-sequence
path), **59** and **83** (the MoE expert-tile rule, under the new `RDNA4_M` clause in the same `if`), and
**77** (the exact-verify attention branch, now beside `flash_attn_ext_rows`). Six more - 80 and 85-89 -
were cascade rejects that applied unchanged once their prerequisites landed. 31 is also the one that
taught the build gate its job: it applied at zero fuzz and zero offset and then failed to compile,
because the kernel it launches had grown a parameter.

Five patches needed real authoring at this ref, not a re-cut: **33** (re-anchored standalone after 32
was deleted, and now carrying the `sched_backend_is_cpu` helper 32 used to define), **35** (the MMVQ
batch-size function was split upstream), **42** (the qwen4exp graph constructors lost their `model`
parameter when it became a member of `llm_graph_context`), **71** (upstream's `mm_ids_helper` swapped
its `bounds` output for a `tile_spec` list, so the port computes the prefix sums itself) and **83**
(its `mmq.cu` hunk is dropped with 59). 71 is the one that applied at zero fuzz and zero offset and
then failed to compile - the second time the build gate has caught what the apply gates cannot.

Patch 24 stays opt-in here, for the reproducibility reason above; that decision was taken at `v0.9.0`
and measured, and nothing at `b12211` changes it. The `v0.9.0` tag is that release - 78 files, patch 24
off - and this branch is the next ref, not a revision of it.
