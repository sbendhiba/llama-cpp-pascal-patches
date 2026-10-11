# llama-cpp-pascal-patches

43 performance patches for [llama.cpp](https://github.com/ggml-org/llama.cpp), developed and
measured on Tesla P100 (GP100, `sm_60`). Generated against llama.cpp **`v0.6.0`**, tagged `v0.6.0`
here. The `b11539` branch trails a llama.cpp prerelease and has **not been measured on a card**;
this branch is the one to use unless you are chasing that specific build.

Unofficial, and unaffiliated with the llama.cpp project. MIT, matching llama.cpp.

GP100 is the reason any of this exists: alone among Pascal chips it has no `DP4A` and has
full-rate `HFMA2`. So llama.cpp's quantized matmul paths fall back to emulation and cuBLAS, while
the one instruction the card is genuinely good at goes unused. The patches named `sm60` exploit
that asymmetry — but only 8 of the 43 are gated to Pascal-era hardware. 34 are not
hardware-scoped at all, and one (07) changes an unconditional constant every GPU sees.

## Apply

Both trees at the **same ref**. Released llama.cpp versions are tags in this repo (`git tag -l`
lists them, e.g. `v0.6.0`); llama.cpp prereleases and work in progress are branches, like `b11539`.

```sh
# 1. the patch set, at the ref you are building against
git clone https://github.com/sbendhiba/llama-cpp-pascal-patches
cd llama-cpp-pascal-patches
git checkout v0.6.0        # a prerelease would be a branch: git checkout b11539
cd ..

# 2. llama.cpp at that same ref
git clone --branch v0.6.0 https://github.com/ggml-org/llama.cpp
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

Whole set against stock, **same llama.cpp tag on both arms**, on 2×Tesla P100-PCIE-16GB, `-fa on
-ub 2048 -t 28`, 5 reps per cell:

| Model | Split | pp t/s | | tg t/s | |
|---|---|---:|---:|---:|---:|
| Ornith-1.0-35B-A3B UD-Q4_K_M (`-p 4096 -n 128`) | `-sm tensor` | 1087.5 → 1080.5 | **−0.6%** | 56.23 → 61.07 | **+8.6%** |
| Ornith-1.0-35B-A3B UD-Q4_K_M (`-p 4096 -n 128`) | `-sm layer` | 890.3 → 927.3 | **+4.2%** | 55.28 → 70.78 | **+28.0%** |
| Qwen3.8-27B Q4_0 dense (`-p 2048 -n 128`) | `-sm tensor` | 369.9 → 368.1 | **−0.5%** | 23.30 → 30.05 | **+29.0%** |

Both arms are build line `d81235049 (11429)` = **`v0.6.0`**, the tree this branch ships - stock from a
pristine checkout, patched from the tree after the apply loop - so the version is not inside the
result. The largest standard deviation in the table is 2.9 t/s (0.3% of its own mean), so the decode
gains are not noise. Raw rows with σ per cell are in
[benchmarks/results/v0.6.0-p100x2-summary.md](benchmarks/results/v0.6.0-p100x2-summary.md).

Two things in that table are worth understanding before you reuse it:

- **`-sm tensor` runs without patch 22.** A tensor split is a meta device, and 22 detects that and
  disables its decode graph slots, so the +8.6% and +29.0% rows are the other 42 patches - what you
  get when 22's VRAM cost or the meta conflict rules it out. It is the guard doing it, not luck:
  +29.0% reproduces an earlier build with 22 physically removed, to within 0.1%. `-sm layer` keeps
  both devices real, so slots stay live there, and that row shows **+28.0%** decode and +4.2%
  prompt.
- **On two cards, `-sm layer` is where this set pays.** Patched decode is 16% higher under layer
  than tensor (70.78 vs 61.07) while stock is within 2% (55.28 vs 56.23). That is not a clean A/B of
  22 - the split strategies differ in their own right - but most of the gap is slots. If you picked
  `-sm` by instinct, pick it again.

**What those rows do not cover.** They were run by hand rather than with
[`benchmarks/run.sh`](benchmarks/run.sh), so the arms were not interleaved and neither temperature
nor environment was recorded; treat the two small pp regressions as "neutral" rather than as a
measured loss. There is no output-identity or perplexity check beside them yet. And they are
two-card numbers - single-card behaviour is still described by the older figures below, which are
themselves old.

For comparison, on one P100, end-to-end decode from `llama-server`, geometric mean over three
prompts, MTP speculation and a realistic sampler:

| Model | Stock | Patched | |
|---|---:|---:|---:|
| Qwen3.5 9B dense (Q4_1, MTP head Q4_1) | 76.99 t/s | **136.11 t/s** | **+76.8%** |
| Qwen3.6 35B-A3B MoE (Q2_K_XL, dense layers Q5_1) | 67.01 t/s | **120.57 t/s** | **+79.9%** |
| Qwen3.8-Flash-Next MoE (UD-IQ3_XXS, q4_0 KV, ctx 262,144, 40k deep) | 8.63 t/s | **20.47 t/s** | **+137%** |

**Read the vintage before you quote these.** Rows 1 and 2 were taken on `b10133`, row 3 on
`v0.4.0`, and neither has been re-taken against the tree that now ships. Each rebase since has been
measured against the previous patched tree rather than against stock, so these are several unmeasured
rebases away from what actually runs today. The v0.4.0-to-v0.5.0 delta was +3.4% and +1.5% on
prompt processing, flat to +1.0% on decode. Full provenance in
[docs/benchmarking.md](docs/benchmarking.md).

Unless a patch says otherwise its output is **bit-identical** to the unpatched build. The ones that
change output are listed in [docs/patches.md](docs/patches.md) with the evidence, and most of them
are off by default behind an environment variable.

## Layout

```
patches/            43 .patch files, applied in numeric order
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
  building is not the same as measuring.
