# llama-cpp-pascal-patches

Performance patches for [llama.cpp](https://github.com/ggml-org/llama.cpp), developed and
measured on Tesla P100 (GP100, `sm_60`). This branch is the **`v0.7.0`** set (41 patches); every
llama.cpp ref this set tracks gets its own branch in this repo, and released ones are tagged too
(`git tag -l`).

**Status of this `v0.7.0` branch:** all 41 patches are regenerated against llama.cpp **`v0.7.0`**
(`620d2225c`) and verified to apply in numeric order at zero fuzz and zero offset, with every
identifier they introduce resolving in the patched tree. Stock and patched trees at that commit have
both been **built** at `sm_60` under CUDA 12.4 with zero errors - so this set compiles on the card -
but it has **not been benchmarked** at this ref. Every number below is inherited from `v0.6.0` and
belongs to those builds, not to this one. The porting notes - one patch dropped because upstream
absorbed it, one patch brought back from parked, and one deliberate divergence in patch 22 - are at
the top of [`docs/patches.md`](docs/patches.md).

Unofficial, and unaffiliated with the llama.cpp project. MIT, matching llama.cpp.

GP100 is the reason any of this exists: alone among Pascal chips it has no `DP4A` and has
full-rate `HFMA2`. So llama.cpp's quantized matmul paths fall back to emulation and cuBLAS, while
the one instruction the card is genuinely good at goes unused. The patches named `sm60` exploit
that asymmetry — but only 8 of them are gated to Pascal-era hardware, and this branch drops one of
those 8 (08, absorbed by upstream). The rest are not hardware-scoped at all, and one (07) changes an
unconditional constant every GPU sees.

## Apply

Both trees at the **same ref**. llama.cpp releases are tagged in this repo (`git tag -l` lists them,
e.g. `v0.6.0`, `v0.7.0`); prereleases and work in progress get a branch named after the build number.

```sh
# 1. the patch set, at the ref you are building against
git clone https://github.com/sbendhiba/llama-cpp-pascal-patches
cd llama-cpp-pascal-patches
git checkout v0.7.0        # or a prerelease branch, e.g: git checkout b11601
cd ..

# 2. llama.cpp at that same ref
git clone --branch v0.7.0 https://github.com/ggml-org/llama.cpp
cd llama.cpp

# 3. apply in numeric order, stop at the first failure
for p in ../llama-cpp-pascal-patches/patches/*.patch; do
  patch -p1 -F0 < "$p" || { echo "FAILED: $p"; break; }
done
```

Checking out a tag leaves you detached on purpose: that tree is the one the measurements describe. A
patch set applied to a different llama.cpp ref is not this set - the patches are generated at exact
context, so expect rejects rather than a partial success.

Apply in numeric order and **keep `-F0`**. Fuzzy application is the dangerous failure mode here:
it can succeed while silently landing a hunk in the wrong place. Stop at the first failure rather
than patching onto a half-patched tree. If you copied this repo onto Windows, the patch files may
carry CRLF; pipe through `sed -e 's/\r$//'` if `patch` complains.

The set is not all-or-nothing, but you cannot pick freely either: several patches touch the same
lines and later ones build on earlier ones. See
[RELEASING.md](RELEASING.md#dropping-a-single-patch) for how to find a patch's real dependency set.

Then build for Pascal: `cmake -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 ...`. CUDA **12.x** —
CUDA 13 dropped Pascal code generation. nixpkgs' default `cudaCapabilities` starts at 7.5, which
leaves no `sm_60` code in the binary and fails at runtime with *"named symbol not found"*.

## What to expect

Every patch carries its measurement in the comments it adds to the source, and
[docs/patches.md](docs/patches.md) is the index: per patch, the scope tag, the effect, the kill
switch, what it changes about the output, and the alternatives that were tried and rejected.

### Scripted - 2026-10-11, `v0.6.0`, 2x Tesla P100-PCIE-16GB

Four configurations through `benchmarks/run.sh`, 5 rounds each, arms interleaved with the order
reversed on even rounds. Both arms are build line `d81235049 (11429)` = `v0.6.0`: stock from a
pristine checkout, patched from the tree after the apply loop. `-fa on -t 28 -ub 2048 -n 128`, with
`-p 512,2048` on the 27B and `-p 4096` on the 35B. The card was waited back down to 45 C between
runs, so each run starts cool and drifts within its own range: 38-60, 44-56, 44-55 and 44-53 C, in
table order.

| Model | Split | pp t/s | | tg 128 t/s | | sd across rounds |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.8-27B Q4_0 dense | `-sm tensor` | 380.06 → 378.13 | **−0.5%** | 23.22 → **30.14** | **+29.8%** | 0.05 / 0.07 |
| Qwen3.8-27B Q4_0 dense | `-sm layer` | 219.50 → 223.53 | **+1.8%** | 15.26 → **21.14** | **+38.5%** | 0.00 / 0.02 |
| Ornith-1.0-35B-A3B UD-Q4_K_M | `-sm tensor` | 1091.82 → 1083.73 | **−0.7%** | 58.27 → **61.60** | **+5.7%** | 0.90 / 1.10 |
| Ornith-1.0-35B-A3B UD-Q4_K_M | `-sm layer` | 914.89 → 953.05 | **+4.2%** | 55.26 → **70.64** | **+27.8%** | 0.02 / 0.10 |

The 27B was also taken at pp512: 229.72 → 229.07 (−0.3%) under tensor, 145.47 → 145.20 (−0.2%) under
layer, and once more at llama-bench's default `-ub 512` (+29.44% decode, in
`v0.6.0-p100x2-tensor-27b-ub512.csv`) - the decode gain survives the batch size, the prefill number
does not (225.00 against 380.06 t/s). One row per arm, round and test - with build commit, temperature
and environment - is in [benchmarks/results/](benchmarks/results/) as
`v0.6.0-p100x2-tensor-27b.csv`, `-layer-27b`, `-tensor-35b` and `-layer-35b`, each with the table
`run.sh` printed beside it.

Three things to read out of that table, and the middle one is a correction:

- **The set is a decode set.** Prefill is flat to mildly negative on tensor split and positive on
  layer (+1.8% and +4.2%), which is what 22 and the scheduler patches do to the reserve path.
- **`-sm layer` is where patch 22 runs, so its gains are larger - but that does not decide which
  split to use.** A tensor split is a meta device and 22 zeroes its decode slots, so the tensor rows
  are the set minus 22. Under layer both devices stay real and 22 is live, and the decode gain is
  bigger in both models: +38.5% against +29.8% on the dense 27B, +27.8% against +5.7% on the MoE. The
  absolute answer is different for each. On the MoE, layer wins outright (70.64 against 61.60). On
  the dense 27B, **tensor is still faster** - 30.14 against 21.14 - because stock layer starts far
  lower (15.26 against 23.22) and +38.5% of a low base does not catch up. Pick the split on measured
  throughput for your model, then expect the patches to help more on layer than on tensor.
- **The MoE tensor gain is smaller than the hand-run number suggested**: +5.7% here against the +8.6%
  below. The stock arm moved, not the patched one (58.27 here against 56.23 there, with about 1 t/s of
  spread on each), so the older figure was flattered by a slow control. Quote +5.7%.

### By hand - same tag, `llama-bench` direct, `-ub 2048`

Whole set against stock on the same two cards, 5 reps per cell, taken before the script was usable:

| Model | Split | pp t/s | | tg t/s | |
|---|---|---:|---:|---:|---:|
| Ornith-1.0-35B-A3B UD-Q4_K_M (`-p 4096 -n 128`) | `-sm tensor` | 1087.5 → 1080.5 | **−0.6%** | 56.23 → 61.07 | **+8.6%** |
| Ornith-1.0-35B-A3B UD-Q4_K_M (`-p 4096 -n 128`) | `-sm layer` | 890.3 → 927.3 | **+4.2%** | 55.28 → 70.78 | **+28.0%** |
| Qwen3.8-27B Q4_0 dense (`-p 2048 -n 128`) | `-sm tensor` | 369.9 → 368.1 | **−0.5%** | 23.30 → 30.05 | **+29.0%** |

Arms were not interleaved and no temperature or environment was recorded; σ per cell, the arm
definitions and the list of what is missing are in
[benchmarks/results/v0.6.0-p100x2-manual.md](benchmarks/results/v0.6.0-p100x2-manual.md). Two of these
three rows survived the scripted re-measurement and one did not, which is why the scripted table comes
first: layer-35b agrees to a quarter of a point with a stock arm identical to 0.02 t/s, tensor-27b now
has three runs agreeing, and tensor-35b fell from +8.6% to +5.7%.

**Read the vintage before you quote any of this.** Both tables above are the `v0.6.0` set (46 files),
not the 41 this branch ships, and nothing on this branch has been run on a card - it only builds. That
matters in both directions: those rows were produced with five patches (08, 34, 39, 46, 48) that this
branch no longer contains, because upstream absorbed or obsoleted them
([docs/patches.md](docs/patches.md)), and the 41 that survived were regenerated against a different
tree. No output-identity check exists beside any of it.

For comparison, the older single-P100 figures, end-to-end decode from `llama-server`, geometric mean
over three prompts, MTP speculation and a realistic sampler:

| Model | Stock | Patched | |
|---|---:|---:|---:|
| Qwen3.5 9B dense (Q4_1, MTP head Q4_1) | 76.99 t/s | **136.11 t/s** | **+76.8%** |
| Qwen3.6 35B-A3B MoE (Q2_K_XL, dense layers Q5_1) | 67.01 t/s | **120.57 t/s** | **+79.9%** |
| Qwen3.8-Flash-Next MoE (UD-IQ3_XXS, q4_0 KV, ctx 262,144, 40k deep) | 8.63 t/s | **20.47 t/s** | **+137%** |

Rows 1 and 2 were taken on `b10133`, row 3 on `v0.4.0`, and none has been re-taken since: each
rebase was measured against the previous patched tree rather than against stock, so they are several
unmeasured rebases away from what runs today. The v0.4.0-to-v0.5.0 delta was +3.4% and +1.5% on
prompt processing, flat to +1.0% on decode. Full provenance in
[docs/benchmarking.md](docs/benchmarking.md).

Unless a patch says otherwise its output is **bit-identical** to the unpatched build. The ones that
change output are listed in [docs/patches.md](docs/patches.md) with the evidence, and most of them
are off by default behind an environment variable.

## Layout

```
patches/            41 .patch files, applied in numeric order
docs/patches.md     per-patch index: scope, effect, kill switch, output changes, dead ends
docs/benchmarking.md  the measurement discipline this set needed - thermal control, noise floors,
                      why two microbenchmark-driven predictions were wrong
benchmarks/         how to re-measure, and the results of each release
RELEASING.md        what to do when llama.cpp ships a new version
CONTRIBUTING.md     adding a patch, reporting a measurement, what a good PR looks like
```

## Caveats

- Tuned for **decode**: single-stream decode with a speculative verify batch of 2-5. Large-batch
  serving was not a target.
- Everything was measured on **Pascal** (`sm_60`), on one card or on two. Patches scoped to `sm_60`,
  `pre-Volta` or `pre-Turing` change values upstream tuned for newer hardware, and `pre-Turing`
  includes Volta, which was never measured. Patch 07 is not gated at all. Measure before using any
  of them on Turing and later.
- `patch 22 decode-sched-slots` conflicts with the meta backend and **disables itself** under
  `-sm tensor`. Multi-GPU with `-sm layer` does run it, and that is now measured; the `-sm tensor`
  rows in [What to expect](#what-to-expect) are the whole set minus 22. Single-card numbers all
  include it.
- `CUDA (delta-net)` patches only fire on models with a gated delta-net block (Qwen3-Next, Qwen3.5).
  Elsewhere they never fire, and the only cost is the per-graph pattern match.
- Nothing in this repo compiles the CUDA sources. Applying cleanly is not the same as building, and
  building is not the same as measuring. The `v0.6.0` patched tree *has* been built and run on a P100
  box - that is what every scripted row above is - but `test-backend-ops` has never been run against
  it, and no `-npl` or perplexity number exists beside any of these gains, so the CUDA patches are
  verified by throughput and by eye rather than by an op-level correctness suite. The `v0.7.0` arms
  are further behind: stock and patched both build at `sm_60` at that commit, and neither has been
  run at all.
