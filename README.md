# llama-cpp-pascal-patches

44 performance patches for [llama.cpp](https://github.com/ggml-org/llama.cpp), developed and
measured on Tesla P100 (GP100, `sm_60`). Generated against llama.cpp **`v0.6.0`**, tagged `v0.6.0`
here. The `b11539` branch trails a llama.cpp prerelease and has **not been measured on a card**;
this branch is the one to use unless you are chasing that specific build.

Unofficial, and unaffiliated with the llama.cpp project. MIT, matching llama.cpp.

GP100 is the reason any of this exists: alone among Pascal chips it has no `DP4A` and has
full-rate `HFMA2`. So llama.cpp's quantized matmul paths fall back to emulation and cuBLAS, while
the one instruction the card is genuinely good at goes unused. The patches named `sm60` exploit
that asymmetry — but only 9 of the 44 are gated to Pascal-era hardware. 34 are not
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
layer. One row per arm, round and test - with build commit, temperature and environment - is in
[benchmarks/results/](benchmarks/results/) as `v0.6.0-p100x2-tensor-27b.csv`, `-layer-27b`,
`-tensor-35b` and `-layer-35b`, each with the table `run.sh` printed beside it.

Three things to read out of that table, and the middle one corrects a claim this README used to make:

- **The set is a decode set.** Prefill is flat to mildly negative on tensor split and positive on
  layer (+1.8% and +4.2%), which is what 22 and the scheduler patches do to the reserve path.
- **`-sm layer` is where patch 22 runs, so its gains are larger - but that does not decide which
  split to use.** A tensor split is a meta device and 22 zeroes its decode slots, so the tensor rows
  are the set minus 22. Under layer both devices stay real and 22 is live, and the decode gain is
  bigger in both models: +38.5% against +29.8% on the dense 27B, +27.8% against +5.7% on the MoE. The
  absolute answer is different for each, though. On the MoE, layer wins outright (70.64 t/s against
  61.60) — **but only while both arms use the fallback**; switch the internal AllReduce on and the
  tensor arm beats it, see the next section. On the dense 27B, **tensor is still faster** - 30.14
  against 21.14 - because stock layer
  starts far lower (15.26 against 23.22), and +38.5% of a low base does not catch up. So the larger
  percentage is not a recommendation: pick the split on measured throughput for your model, then
  expect the patches to help more on layer than on tensor.
- **The MoE tensor gain is smaller than the hand-run number suggested**: +5.7% here against the +8.6%
  below. The stock arm moved, not the patched one (58.27 here against 56.23 there, with about 1 t/s of
  spread on each), so the older figure was flattered by a slow control. Quote +5.7%.

An output-identity check (`-npl` or perplexity) is still missing beside all of this; throughput alone
cannot tell a win from a different-but-still-correct answer. That gap is widest for patch 49, whose
whole job is to run a different reduction — see its entry in
[docs/patches.md](docs/patches.md). A fifth scripted run of the same
27B tensor case at llama-bench's default `-ub 512` gave +29.44% decode, so the decode gain is not an
artefact of the batch size - its prefill numbers (225.00 against 380.06 t/s at pp2048) are, which is
why `-ub` belongs in the row rather than assumed.

### Scripted - 2026-10-11, the same cards with the internal AllReduce switched on

Everything above runs both arms through the meta-backend's f32-through-host fallback, because on a
P100 `49-allreduce-multigpu` refuses to run unless asked. Asked, it is by far the largest single
effect in the set. Same runner, same two arms, 5 interleaved rounds, `-sm tensor`, with the session
exporting `GGML_CUDA_AR_PASCAL=1 GGML_CUDA_AR_H2D_STREAM=1` — the stock binary has no such variables,
so it is unaffected and the pair stays like-for-like:

| Model | pp t/s | | tg128 t/s | |
|---|---:|---:|---:|---:|
| Qwen3.8-27B Q4_0 dense (`-p 2048`) | 382.39 → **434.27** | **+13.6%** | 23.27 → **33.86** | **+45.5%** |
| Qwen3.8-27B Q4_0 dense (`-p 512`) | 230.07 → **248.02** | **+7.8%** | — same call, already shown | |
| Ornith-1.0-35B-A3B UD-Q4_K_M (`-p 4096`) | 1091.30 → **1198.96** | **+9.9%** | 58.42 → **70.90** | **+21.4%** |

Three things follow from it, and each changes something written elsewhere in this repo:

- **The split recommendation flips on the MoE.** Patched `-sm tensor` with the internal AllReduce is
  1198.96 pp / 70.90 tg against patched `-sm layer`'s 953.05 / 70.64 — same decode, 26% more
  prefill. The table in the previous section prefers `layer` on the 35B only because the tensor arm
  there was reduced through the host. The dense 27B still prefers `tensor`, with or without it.
- **The two variables are worth +12.6% decode and +14.1% prefill over the same binary with them
  unset** (33.86 against 30.07 and 434.27 against 380.47 at `-p 2048`), which is the increment this
  patch adds on top of the ~29% the rest of the set already delivers. qr0n claimed +10.1% prompt /
  +9.6% decode for the Pascal gate alone at `b10660`; measured here it is larger, and the H2D stream
  has not yet been separated from it. On the MoE the same comparison crosses sessions (70.90 against
  the published 61.60 tensor decode, +15.1%), which is defensible only because the two stock arms
  agree to within 0.3% — it is not an interleaved result.
- **None of it is a correctness result.** With the gate on, the reduction genuinely is a different
  one than the fallback's: no `-npl`, no perplexity, no token-level comparison exists beside these
  rows. Patch 50's history is the warning to take seriously — upstream's first attempt at exactly
  this stream split measured "just as fast" and returned perplexity **7.68 instead of 3.53**.

The patched arm ran warmer than stock inside these sessions (52-62 C against 44-62 C), so the
gains are if anything understated. Provenance per row in
[`benchmarks/results/v0.6.0-p100x2-tensor-27b-ar49on.csv`](benchmarks/results/v0.6.0-p100x2-tensor-27b-ar49on.csv)
and
[`…-tensor-35b-ar49on.csv`](benchmarks/results/v0.6.0-p100x2-tensor-35b-ar49on.csv). The two
`…-aroff` files beside them are the same two tensor configs re-taken after patches 49, 50 and the
N-GPU port were folded into one, which is how the merge was shown to be free at two cards: 30.14 →
30.07 and 70.64 → 70.66 decode, with both arms of the 27B drifting the same way.

**The Pascal gate is now on by default**, so a pre-Volta tensor split gets that table without being
asked; `GGML_CUDA_AR_H2D_STREAM` and `GGML_CUDA_AR_NGPU` are still off. The gate is the one patch in
this set that is not output-neutral, so the flip was made on
[`benchmarks/results/v0.6.0-p100x2-tensor-27b-perplexity.md`](benchmarks/results/v0.6.0-p100x2-tensor-27b-perplexity.md):
wikitext-2 on the 27B, one binary, four arms — internal 5.2524 / 5.2487, fallback 5.2360 / 5.2603,
the no-reduction `layer` control 5.2538 / 5.2427. All within 0.024 of each other (0.46%), the sign
flipping with the ubatch, and the internal path the closer one to the control at both sizes. That
clears "wrong" (qr0n's broken build was 7.68 against 3.53); it does not clear "checked", because no
logit-level comparison exists yet. Reproduce it with [`benchmarks/ppl.sh`](benchmarks/ppl.sh). 

### By hand - same tag, `llama-bench` direct, `-ub 2048`

Whole set against stock on the same two cards, 5 reps per cell, taken before the script was usable:

| Model | Split | pp t/s | | tg t/s | |
|---|---|---:|---:|---:|---:|
| Ornith-1.0-35B-A3B UD-Q4_K_M (`-p 4096 -n 128`) | `-sm tensor` | 1087.5 → 1080.5 | **−0.6%** | 56.23 → 61.07 | **+8.6%** |
| Ornith-1.0-35B-A3B UD-Q4_K_M (`-p 4096 -n 128`) | `-sm layer` | 890.3 → 927.3 | **+4.2%** | 55.28 → 70.78 | **+28.0%** |
| Qwen3.8-27B Q4_0 dense (`-p 2048 -n 128`) | `-sm tensor` | 369.9 → 368.1 | **−0.5%** | 23.30 → 30.05 | **+29.0%** |

Arms were not interleaved and no temperature or environment was recorded, so the pp regressions read
as "neutral" rather than as measured losses; σ per cell, the arm definitions and the list of what is
missing are in
[benchmarks/results/v0.6.0-p100x2-manual.md](benchmarks/results/v0.6.0-p100x2-manual.md).

Two of these three rows survived the scripted re-measurement and one did not, which is the whole
reason the scripted tables come first. Layer on the 35B agrees to a quarter of a point (+27.8%
against +28.0%) and to within 0.05 on prefill (+4.2% both times), with a near-identical stock arm
(55.26 against 55.28). Tensor on the 27B now has three independent runs agreeing - +29.0% here,
+29.44% and +29.81% scripted. Tensor on the 35B is the one that moved, down to +5.7%.

### By hand - a qwen4exp MoE that does not fit the cards, `-p 512 -ub 2048`

The same two cards and the same tag, against `Qwen3.8-Flash-Next-GSQ-RCO-Coder` IQ1_M (llama-bench
calls it `qwen4exp A3B IQ1_M - 1.75 bpw`, 116.51 B params, 54.39 GiB):

| Split | pp t/s | | tg t/s | |
|---|---:|---:|---:|---:|
| `-sm tensor` | 365.20 → 346.72 | **−5.1%** | 24.91 → 28.18 | **+13.1%** |
| `-sm layer` | 259.13 → 243.43 | **−6.1%** | 21.58 → 25.96 | **+20.3%** |

Two firsts in one run: the patched arm here is the 44-patch set with the merged AllReduce patch, and
this is the only model where the set loses prompt processing by more than noise. With the AllReduce
variables unset the patched arm refuses the internal reduction on Pascal and takes the same fallback
stock does, so 49 is not what causes it — the loss is somewhere in the other 43, on the one
architecture in this set with model-specific patches of its own (42, 46, 47), and no kill-switch A/B
has been run to locate it. The file records the third arm (`b11539`, which has no AllReduce patch and
no stock control either), the two transcript cells that arrived corrupted, and why `-p 512` on a
54 GiB model makes this row nobody else's comparison:
[benchmarks/results/v0.6.0-p100x2-qwen4exp-manual.md](benchmarks/results/v0.6.0-p100x2-qwen4exp-manual.md).

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
patches/            44 .patch files, applied in numeric order
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
- Prompt processing is where this set is weakest on **qwen4exp**: measured once, on a 116 B A3B
  MoE at `-p 512`, the whole set costs **−5.1% (tensor) / −6.1% (layer)** while buying +13.1% /
  +20.3% decode. Every other model in [What to expect](#what-to-expect) is flat to +4% on prefill.
  The responsible patch has not been identified, so if prefill is what you care about on that
  architecture, measure before you deploy. One way out may be patch 49's gate: on the two models
  where it has been measured the tensor split gains 8-14% prefill, which would more than cover a 5%
  loss — but that row was taken with the gate **off**, before it became the default, so qwen4exp has
  still never seen it.
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
- Patch **49 `allreduce-multigpu` is ported from another repo**
  ([qr0n/p100-llama-cpp](https://github.com/qr0n/p100-llama-cpp), MIT), not written here. Its Pascal
  gate is **on by default** — the only part of this set that is not output-neutral without saying so
  in the code — and the two other variables are off: `GGML_CUDA_AR_H2D_STREAM=1` puts the copy-engine
  return leg on a second stream, `GGML_CUDA_AR_NGPU=1` allows more than two devices.
  `GGML_CUDA_AR_PASCAL=0` restores upstream's behavior exactly. The gate changes which reduction runs
  across a tensor split, so it can change output; it was flipped on the strength of a paired
  perplexity check whose widest spread is 0.46% (see
  [What to expect](#what-to-expect)), and the N-GPU half was validated by its author on *virtual*
  devices, not on three or four cards. It compiles with nvcc 12.4 at `sm_60`, and on two real cards
  the gate is worth **+13.6% pp / +45.5% decode on the dense 27B and +9.9% / +21.4% on the 35B MoE**
  against the same arms on the fallback — better than qr0n's +10.1% / +9.6% at `b10660`, with the
  stream still not separated out of it. One part of it is **not** behind any variable: making the
  reduction N-way rewrites both kernels, and that rewrite runs at two cards too, where it is meant to
  be arithmetic-identical (`w0 + w1` is what `own + peer` already was). It was re-taken to prove that:
  30.14 → 30.07 and 70.64 → 70.66 decode against the pre-merge build.
- Nothing in this repo compiles the CUDA sources. Applying cleanly is not the same as building, and
  building is not the same as measuring. The `v0.6.0` patched tree *has* been built and run on a P100
  box - that is what every scripted row above is - but `test-backend-ops` has never been run against
  it, and no `-npl` or perplexity number exists beside any of these gains. The CUDA patches are
  therefore verified by throughput and by eye, not by an op-level correctness suite.
