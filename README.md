# llama-cpp-pascal-patches

39 performance patches for [llama.cpp](https://github.com/ggml-org/llama.cpp), developed and
measured on Tesla P100 (GP100, `sm_60`). Generated against llama.cpp **`b11539`** (`10a60cf30`);
the `v0.6.0` set is tagged `v0.6.0` in this repo.

Unofficial, and unaffiliated with the llama.cpp project. MIT, matching llama.cpp.

GP100 is the reason any of this exists: alone among Pascal chips it has no `DP4A` and has
full-rate `HFMA2`. So llama.cpp's quantized matmul paths fall back to emulation and cuBLAS, while
the one instruction the card is genuinely good at goes unused. The patches named `sm60` exploit
that asymmetry — but only 8 of the 39 are gated to Pascal-era hardware. The other 30 are not
hardware-scoped at all, and one (07) changes an unconditional constant every GPU sees.

## Apply

Both trees at the **same ref**. Released llama.cpp versions are tags in this repo (`git tag -l`
lists them, e.g. `v0.6.0`); llama.cpp prereleases and work in progress are branches like this one.

```sh
# 1. the patch set, at the ref you are building against
git clone https://github.com/sbendhiba/llama-cpp-pascal-patches
cd llama-cpp-pascal-patches
git checkout b11539        # a released ref would be: git checkout v0.6.0
cd ..

# 2. llama.cpp at that same ref
git clone --branch b11539 https://github.com/ggml-org/llama.cpp
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

`benchmarks/run.sh`, 5 rounds, arms interleaved with the order reversed on even rounds, both arms
build line `d81235049 (11429)` = `v0.6.0`: stock from a pristine checkout, patched from the tree
after the apply loop. `-fa on -sm tensor -t 28 -p 512,2048 -n 128`, so llama-bench's default
`-ub 512`. Card temperature: 38 C at the first round, 62 C at the last.

| Model | test | stock | patched | Δ | sd across rounds |
|---|---|---:|---:|---:|---:|
| Qwen3.8-27B Q4_0 dense | pp512 | 230.00 | 229.11 | **−0.4%** | 0.52 / 0.35 |
| Qwen3.8-27B Q4_0 dense | pp2048 | 225.00 | 228.95 | **+1.8%** | 0.18 / 0.18 |
| Qwen3.8-27B Q4_0 dense | tg128 | 23.25 | **30.10** | **+29.4%** | 0.06 / 0.06 |

Every row is in [benchmarks/results/v0.6.0-p100x2.csv](benchmarks/results/v0.6.0-p100x2.csv) with its
round number, build commit, temperature and environment; the table above is
[benchmarks/results/v0.6.0-p100x2-summary.md](benchmarks/results/v0.6.0-p100x2-summary.md). That
+29.4% is the whole set **minus patch 22**, which switches itself off on a meta device - the first
bullet under the hand-run table below explains why that is the interesting caveat rather than a
footnote. Still missing: a second model, `-ub 2048` to match the prefill numbers below, and an
output-identity check beside it.

### By hand - same tag, `llama-bench` direct, `-ub 2048`

Whole set against stock on the same two cards, 5 reps per cell, taken before the script was usable:

| Model | Split | pp t/s | | tg t/s | |
|---|---|---:|---:|---:|---:|
| Ornith-1.0-35B-A3B UD-Q4_K_M (`-p 4096 -n 128`) | `-sm tensor` | 1087.5 → 1080.5 | **−0.6%** | 56.23 → 61.07 | **+8.6%** |
| Ornith-1.0-35B-A3B UD-Q4_K_M (`-p 4096 -n 128`) | `-sm layer` | 890.3 → 927.3 | **+4.2%** | 55.28 → 70.78 | **+28.0%** |
| Qwen3.8-27B Q4_0 dense (`-p 2048 -n 128`) | `-sm tensor` | 369.9 → 368.1 | **−0.5%** | 23.30 → 30.05 | **+29.0%** |

Arms were not interleaved and no temperature or environment was recorded, so the two small pp
regressions read as "neutral" rather than as a measured loss; σ per cell and the list of what is
missing are in
[benchmarks/results/v0.6.0-p100x2-manual.md](benchmarks/results/v0.6.0-p100x2-manual.md). The decode
column now has the scripted check above agreeing to within half a point; the prefill columns of the
two tables are not comparable, because llama-bench defaults to `-ub 512` and these passed 2048.

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

**Read the vintage before you quote any of this.** The rows above are the `v0.6.0` set (43 patches),
not the `b11539` set this branch now ships, and the `b11539` rebase has not been measured on a card
at all. That matters in both directions: the table was measured with four patches (34, 39, 46, 48)
that this branch no longer contains, because upstream absorbed or obsoleted them
([docs/patches.md](docs/patches.md)), and the 39 that did survive were regenerated against a
different tree. These three rows were also run by hand rather than with
[`benchmarks/run.sh`](benchmarks/run.sh), so the arms were not interleaved and neither temperature
nor environment was recorded; treat the two small pp regressions as "neutral" rather than as a
measured loss, and there is no output-identity check beside them yet.

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
patches/            39 .patch files, applied in numeric order
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
