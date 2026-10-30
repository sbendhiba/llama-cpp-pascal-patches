# llama-cpp-pascal-patches

Performance patches for [llama.cpp](https://github.com/ggml-org/llama.cpp), developed and
measured on Tesla P100 (GP100, `sm_60`). This branch is the **`v0.8.0`** set (72 patches: the 40 we
carry, plus the 32-patch P60 block on top); the set is tagged per llama.cpp ref (`git tag -l` lists
them - `v0.6.0`, `v0.7.0`, and the prerelease line `b11601`), and the branch is always the newest one.

**Status of this `v0.8.0` set:** all 40 patches are regenerated against llama.cpp **`v0.8.0`**
(`8642541b2`) and verified to apply in numeric order at zero fuzz and zero offset, with every
identifier they introduce resolving in the patched tree. Stock and patched trees at that commit have
both been **built** at `sm_60` under CUDA 12.4 with zero errors, and the set has been through a
**full scripted session** at 160 W - 5 interleaved rounds, four configs, four arms: `v0.7.0` and
`v0.8.0`, stock and patched, on the 27B and the 35B, both splits. The `v0.8.0` set is ahead of the
`v0.7.0` set by 1.1-3.7% on every cell, and upstream `v0.8.0` is within ±2% of `v0.7.0` at that
power. The numbers below the `v0.8.0` heading are this set's; the `v0.7.0` and `v0.6.0` rows are
kept beside them because they are what the gain is compared against. The porting notes - patch 17
gone (upstream `623c16393` now does the same register cache with float4 loads) and 02/09, 19, 42, 50
re-anchored onto restructured upstream code - are at the top of
[`docs/patches.md`](docs/patches.md).

Unofficial, and unaffiliated with the llama.cpp project. MIT, matching llama.cpp.

GP100 is the reason any of this exists: alone among Pascal chips it has no `DP4A` and has
full-rate `HFMA2`. So llama.cpp's quantized matmul paths fall back to emulation and cuBLAS, while
the one instruction the card is genuinely good at goes unused. The patches named `sm60` exploit
that asymmetry — but only 10 of the 40 are gated to Pascal-era hardware. 29 are not
hardware-scoped at all, and one (07) changes an unconditional constant every GPU sees. The P60 block
(52-83) is gated to `sm_60` as a whole.

## Apply

Both trees at the **same ref**. Every ref this set is generated against is a tag in this repo
(`git tag -l` lists them - `v0.6.0`, `v0.7.0`, and the prerelease line `b11601`); `main` is the newest.

```sh
# 1. the patch set, at the ref you are building against
git clone https://github.com/sbendhiba/llama-cpp-pascal-patches
cd llama-cpp-pascal-patches
git checkout v0.8.0        # any ref this set was cut against is a tag: git tag -l
cd ..

# 2. llama.cpp at that same ref
git clone --branch v0.8.0 https://github.com/ggml-org/llama.cpp
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

### Scripted - 2026-10-30, `v0.8.0`, three arms built from the published patch files

The point of this run is provenance: each arm is a tree made by taking llama.cpp `v0.8.0`
(`8642541b2`) and applying **this repo's own `patches/` with `patch -p1 -F0`**, exactly as the
[Apply](#apply) section tells you to, then building it with the same flags.

| arm | tree | apply result |
|---|---|---|
| `stock` | llama.cpp `v0.8.0`, no patches | - |
| `patch1` | the 40 patches as they are **on origin** (`1cd431e`) | 0 rejects, 0 fuzzy hunks, 35 files touched |
| `patch2` | those 40 + the P60 block 52-83 | 0 rejects, 0 fuzzy hunks, 52 files touched |

2x P100 at **160 W**, `-fa on -t 28 -ub 2048 -n 128`, 3 interleaved rounds with the order reversed on
even rounds, cooled to 45 C before each config. Standard deviation across rounds is under 1.4 t/s on
every cell except `stock` tg on the 35B tensor split (1.04). Raw rows:
`benchmarks/results/v0.8.0-clean-160w-p100x2-*.csv`.

| model | split | arm | pp512 t/s | pp2048 t/s | tg128 t/s |
|---|---|---|---:|---:|---:|
| Qwen3.8-27B **UD-Q4_K_M** | `-sm tensor` | stock | 230.10 | 377.84 | 20.99 |
| | | patch1 | 381.36 (+65.7%) | 424.88 (+12.5%) | 29.65 (+41.2%) |
| | | patch2 | 381.77 (+65.9%) | 424.43 (+12.3%) | 29.63 (+41.2%) |
| Qwen3.8-27B UD-Q4_K_M | `-sm layer` | stock | 143.80 | 224.68 | 13.21 |
| | | patch1 | 205.54 (+42.9%) | 223.87 (-0.4%) | 18.22 (+38.0%) |
| | | patch2 | 205.04 (+42.6%) | 223.72 (-0.4%) | 18.19 (+37.7%) |
| Cyber-Tiel-35B-A3B MTP UD-Q4_K_XL | `-sm tensor` | stock | 767.96 | - | 59.78 |
| | | patch1 | 841.37 (+9.6%) | - | 79.61 (+33.2%) |
| | | patch2 | 1107.25 (+44.2%) | - | 86.34 (+44.4%) |
| Cyber-Tiel-35B-A3B MTP UD-Q4_K_XL | `-sm layer` | stock | 488.75 | - | 58.34 |
| | | patch1 | 502.45 (+2.8%) | - | 69.19 (+18.6%) |
| | | patch2 | 721.73 (+47.7%) | - | 77.80 (+33.4%) |

Three things come out of it:

- **The P60 block is free on the dense model.** On the 27B, `patch2` against `patch1` is
  +0.11 / -0.05 / -0.11% (pp512 / tg / pp2048) on tensor and -0.24 / -0.16 / -0.07% on layer - all
  inside the round-to-round spread. That is the question the default-on raises, since the block's
  gates are `sm_60`-wide but its kernels are written for a 2048-wide MoE: on a model it cannot use,
  it now demonstrably costs nothing. (The earlier dense measurement that showed **-1.8% pp512** was
  taken with `GGML_CUDA_P60_MEGA=1` armed, which is not the default.)
- **On the MoE the block is worth a further +31.6% pp512 / +8.5% tg** on tensor and
  **+43.6% / +12.4%** on layer, on top of the 40-patch set. The 35B here is UD-Q4_K_XL with an MTP
  block; the Q4_K_M Ornith row in the section above is the same architecture one quant up.
- This also adds the **UD-Q4_K_M quant of the 27B** to the record. Every earlier 27B row in this
  README is Q4_0; the Q4_K_M gains are the same shape (+65.7% pp512, +41.2% tg on tensor) but the
  absolute numbers are lower (381.36 against Q4_0's 391.37), which is what a K-quant costs on this
  card.

`patch2` with the persistent kernels armed (`GGML_CUDA_P60_MEGA=1`, same build, 3 rounds, 40-42 C):
**1068.62 pp512 / 99.40 tg128** on the 35B layer split - +118.6% and +70.4% over stock. See
the P60-block section above for why that
switch is not on by default.

### Scripted - 2026-10-30, `v0.8.0`, 72 patches, 160 W - **the P60 block on top**

The 32 patches 52-83 are a port of [Mrinohk/qwen36-p100-patches](https://github.com/Mrinohk/qwen36-p100-patches)
- a second, independent P100 effort for the Qwen3.5/3.6 MoE family - rebased onto this set rather
than merged beside it. Their series is 37 commits; 32 landed. Five were dropped as duplicates of what
this repo already carries: `0005` (patches 24/25/27 do the same GDN gather work), `0006` (absorbed by
upstream `1ab7e5ad2`), `0015`/`0016` (patch 51's cuBLAS algorithm choice, which is broader than theirs),
`0017` (patch 15's draft vocabulary) and `0007` (written against the pre-`623c16393` `rms_norm` kernel;
re-authoring it is open). `62` is theirs and is deleted again by `83`: `VDR_Q6_K_Q8_1_MMVQ` became 2 at
this ref and its staged mapping asserts 1.

What they add, in one line: fp16 (`HFMA2`) matrix-vector kernels for the types GP100 can only emulate,
fused with the RMS norm and the residual adds that feed them; then persistent whole-layer-half kernels
with software grid barriers, and fp16 grouped GEMMs for MoE prompt processing.

Two cards (`CUDA_VISIBLE_DEVICES=0,1`), all cards at **160 W**, `-fa on -ub 2048 -p 512 -n 128 -r 2`,
`-t 28` on two cards and `-t 42` on three, Cyber-Tiel-Coder-35B-A3B MTP UD-Q4_K_XL (`qwen35moe`, the
architecture these kernels are written for), build `27b71f310 (11994)`:

| split | cards | arm | pp512 t/s | tg128 t/s |
|---|---:|---|---:|---:|
| `-sm layer` | 2 | the 40-patch set (P60 off) | 504.25 ± 2.11 | 69.50 ± 0.39 |
| | | **+ P60 as shipped (`GGML_CUDA_P60_MMVQ` on)** | **725.97 ± 5.29** | **78.14 ± 0.48** |
| | | + `GGML_CUDA_P60_MEGA=1` | **1075.37 ± 6.43** | **100.05 ± 0.73** |
| `-sm tensor` | 2 | P60 off | 843.91 ± 3.31 | 80.38 ± 1.04 |
| | | P60 + `MEGA=1` | 1596.22 ± 19.47 | 86.93 ± 1.14 |
| `-sm layer` | 3 | P60 off | 503.48 ± 1.94 | 68.38 ± 0.37 |
| | | P60 + `MEGA=1` | 1076.96 ± 13.30 | 99.07 ± 0.69 |
| `-sm tensor` | 3 | P60 off | 847.52 ± 6.16 | 82.15 ± 2.28 |
| | | P60 + `MEGA=1` | 1596.22 ± 23.30 | 85.60 ± 2.33 |

Ornith-1.0-35B-A3B UD-Q4_K_M, same two cards, `-sm tensor -p 4096 -n 128`: **1278.88 → 1934.91 pp4096
(+51.3%)** and 80.09 → 85.44 tg128 (+6.7%). The `P60 off` arm reproduces the published `v0.8.0`
patched row for that cell (1279.20), which is the control that says these arms are comparable.

The third card changes the verdict on nothing: the 35B MoE is bound by the two cards it already had,
and the third adds ~1% to decode (82.15 vs 80.38 tg with P60 off) and nothing to prefill (847.52 vs
843.91). The P60 gain is the same size on two or three cards.

**`GGML_CUDA_P60_MEGA=1` is not the default, and that is the one place this block is not full speed.**
Two findings, both from the MTP path this repo also ships (patch 15):

- with `--spec-type draft-mtp` and both switches on, the 2-6 token verify kernels **crash** -
  `an illegal memory access was encountered` on the second device at the first verify step, under
  `-sm layer` on two cards. Turning off any single sub-switch (`_Q8T`, `_FA`, `_ATTN`,
  `_FUSE_CONV_STATE`, `_GDN_STATE`, `_FUSE_GLU`) does not help; the crash needs both parents.
- with the multi-token kernels refused but the single-token persistent halves still armed, MTP runs
  and is **wrong-slow**: 46.35 / 48.74 t/s against 73.22 / 74.58 with the block off, and draft
  acceptance collapses from 135/278 to 63/310. The persistent halves take over the target model's
  single-token steps and the drafts stop agreeing with them.
- MTP with the shipped default (`MMVQ` on, `MEGA` off) is the best arm measured: 0.700 draft
  acceptance and no error.

So the persistent kernels stay behind `GGML_CUDA_P60_MEGA=1` until the MTP graphs are re-targeted,
and the flag is documented as "plain decode and prefill only".

The other half of their decode speed is unreachable at this ref for a different reason. Their
persistent GDN half identifies the conv-state gather by the node name `blk.N.conv_states`. llama.cpp
v0.8.0 builds that gather unnamed, over a one-row reshape of the cache (`cache_r_l9 (reshaped)`), so
the match fails and every GDN layer falls back. Matching it by cache name instead took the half over,
and greedy decode came out garbled from the second token (`The the ( bullet) —`, stray `β` and `§`) at
both `-sm layer` and with the conv-state and state-gather elisions disabled - so the kernel is
mishandling the new layout, not just the name. That experiment is reverted in `83`; the half matches
only the named graph, exactly as upstream names it today. It is worth ~+25% tg (100.05 here against
125.61 on their own base) and is the largest remaining item in this block.

Accuracy, measured rather than assumed: perplexity over a 40 KB code-and-documents corpus, 5 chunks
of 2048, `-sm layer`, is 2.0636 ± 0.0500 with the block off and 2.0593 ± 0.0499 with `MEGA=1` - the
same direction and about the same size as the difference their own KL gate reports. Greedy output
stays on-topic and coherent with `MEGA=1` (it is not bit-identical to the off arm: the fp16 matvec
replaces an emulated int8 dot, so the first diverging token moves the rest).

### Scripted - 2026-10-28, `v0.8.0`, 40 patches, 160 W - **the 40-patch set, before the P60 block**

Two cards (`CUDA_VISIBLE_DEVICES=0,1`), all three cards at their **160 W** limit because the machine
is on a UPS that trips above 600 W of system draw, 5 rounds per config, **four arms interleaved** -
`v0.7.0` and `v0.8.0`, stock and patched - with the order reversed on even rounds and a cooldown to
45 C before each config. Flags are the published ones: `-fa on -t 28 -ub 2048 -n 128`, `-p 512,2048`
on the 27B and `-p 4096` on the 35B. Build lines `620d2225c (11669)` for both `v0.7.0` arms and
`8642541b2 (11960)` for both `v0.8.0` arms, nothing exported for any arm, all four built with
`-DGGML_CUDA_NCCL=OFF`. Raw rows: `benchmarks/results/v0.8.0-160w-p100x2-{tensor,layer}-{27b,35b}.csv`.

| Model | Split | arm | pp512 t/s | pp2048 t/s | tg128 t/s |
|---|---|---:|---:|---:|
| Qwen3.8-27B Q4_0 dense | `-sm tensor` | `v0.7.0` stock | 233.17 ± 0.58 | 375.59 ± 0.97 | 23.38 ± 0.02 |
| | | `v0.7.0` patched | 386.89 ± 4.11 | 424.66 ± 1.68 | 33.28 ± 0.02 |
| | | `v0.8.0` stock | 234.59 ± 0.41 | 379.83 ± 0.66 | 23.82 ± 0.04 |
| | | `v0.8.0` patched | **391.37 ± 1.59** | **429.16 ± 0.62** | **34.21 ± 0.02** |
| Qwen3.8-27B Q4_0 dense | `-sm layer` | `v0.7.0` stock | 145.79 ± 0.43 | 223.90 ± 0.48 | 15.11 ± 0.02 |
| | | `v0.7.0` patched | 206.38 ± 0.65 | 222.38 ± 0.58 | 20.65 ± 0.01 |
| | | `v0.8.0` stock | 146.45 ± 0.32 | 226.39 ± 0.22 | 15.44 ± 0.01 |
| | | `v0.8.0` patched | **209.26 ± 0.59** | **224.72 ± 0.29** | **21.19 ± 0.01** |

| Model | Split | arm | pp4096 t/s | tg128 t/s |
|---|---|---:|---:|---:|
| Ornith-1.0-35B-A3B UD-Q4_K_M | `-sm tensor` | `v0.7.0` stock | 1100.38 ± 1.21 | 58.00 ± 1.18 |
| | | `v0.7.0` patched | 1256.01 ± 0.64 | 77.60 ± 0.08 |
| | | `v0.8.0` stock | 1119.66 ± 0.51 | 56.72 ± 1.87 |
| | | `v0.8.0` patched | **1279.20 ± 1.57** | **79.71 ± 0.08** |
| Ornith-1.0-35B-A3B UD-Q4_K_M | `-sm layer` | `v0.7.0` stock | 950.75 ± 3.82 | 55.19 ± 0.03 |
| | | `v0.7.0` patched | 987.80 ± 1.15 | 66.31 ± 0.05 |
| | | `v0.8.0` stock | 965.23 ± 1.58 | 57.36 ± 0.04 |
| | | `v0.8.0` patched | **1003.32 ± 0.59** | **68.76 ± 0.02** |

The gains, stock to patched within each ref:

| config | `v0.7.0` set | `v0.8.0` set |
|---|---|---|
| 27B tensor | +65.9% pp512, +13.1% pp2048, +42.3% tg | **+67.9%, +14.3%, +46.3%** |
| 27B layer | +41.6%, -0.7%, +36.7% | **+43.5%, +0.4%, +40.3%** |
| 35B tensor | +14.1% pp4096, +33.8% tg | **+16.3%, +37.4%** |
| 35B layer | +3.9%, +20.2% | **+5.5%, +24.6%** |

Four things to read out of it:

- **The `v0.8.0` set is faster than the `v0.7.0` set on every cell**, by +1.1 to +3.7% at equal
  power among the patched arms. The sets differ by patch 17's deletion, which should point the other
  way, and by the re-anchoring onto new upstream code, so this is the same patches landing
  differently on the new base, not a deliberate gain. Decode is where it is most visible and
  least noisy (27B tensor 34.21 vs 33.28, 35B layer 68.76 vs 66.31).
- **Upstream `v0.8.0` is within ±2% of `v0.7.0` at 160 W** on this hardware - `v0.8.0` stock is
  +0.6/+1.1/+1.9% on 27B tensor, +0.5/+1.1/+2.2% on 27B layer, +1.8% pp4096 / -2.2% tg on 35B
  tensor and +1.5/+3.9% on 35B layer. A 185 W smoke during the port showed `v0.8.0` stock +22% pp512;
  that run had **three** cards visible and is not comparable, which is why it is not quoted.
- **Prefill is what the 160 W cap costs.** Against the 185 W `v0.7.0` session below, stock pp512 is
  flat (233.6 → 233.2) but stock pp2048 lost 3.8% (390.6 → 375.6), and the patched arm lost 4.9%
  pp512 and 5.4% pp2048; decode is unchanged (33.28 in both sessions). So the `v0.7.0` gain reads
  +74.2% pp512 here as +65.9%, and +14.9% pp2048 as +13.1%, while tg stays at +42%. On the layer
  split the 185 W +1.8% pp2048 becomes -0.7%. The 35B MoE is memory-bound and barely notices the
  cap (`v0.7.0` patched: 1256.01 vs 1257.15 pp4096, 77.60 both).
- The 160 W session re-measured the `v0.7.0` 35B tensor stock tg128 at **58.00 ± 1.18** over 5 clean
  rounds, against 55.51 in the 185 W session, which carried one 46.38 outlier among 57-58 rounds.
  The new row is the better estimate of that cell.

### Scripted - 2026-10-15, `v0.7.0`, 41 patches - **the earlier, 185 W session**

The first full scripted session on this ref, and the first one that had 49, 50 and 51 in the set from
the start. Same box, same runner, 5 rounds, arms interleaved with the order reversed on even rounds,
both arms build line `620d2225c (11669)`, nothing exported for either arm, cooled to 45 C between
configs (38-60, 43-56, 43-54 and 42-52 C in table order). `-fa on -t 28 -ub 2048 -n 128`,
`-p 512,2048` on the 27B and `-p 4096` on the 35B, and both arms built with `-DGGML_CUDA_NCCL=OFF`.
One row per arm, round and test is in `benchmarks/results/` as `v0.7.0-p100x2-tensor-27b.csv`,
`-layer-27b`, `-tensor-35b` and `-layer-35b`, each with the summary `run.sh` printed beside it.

| Model | Split | pp512 t/s | | pp2048 t/s | | tg128 t/s | |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B Q4_0 dense | `-sm tensor` | 233.62 → **406.92** | **+74.2%** | 390.63 → **448.95** | **+14.9%** | 23.43 → **33.28** | **+42.0%** |
| Qwen3.8-27B Q4_0 dense | `-sm layer` | 147.86 → **217.44** | **+47.1%** | 228.27 → 232.29 | **+1.8%** | 15.20 → **20.68** | **+36.1%** |

| Model | Split | pp4096 t/s | | tg128 t/s | |
|---|---|---:|---:|---:|---:|
| Ornith-1.0-35B-A3B UD-Q4_K_M | `-sm tensor` | 1101.23 → **1257.15** | **+14.2%** | 55.51 → **77.60** | **+34.3%** † |
| Ornith-1.0-35B-A3B UD-Q4_K_M | `-sm layer` | 967.28 → **995.21** | **+2.9%** | 55.14 → **66.41** | **+20.4%** |

† One stock round on that cell came back at 46.38 t/s against 57.49 / 57.38 / 57.76 / 58.56 for the
other four. Including it, the stock mean is 55.51 and the delta is **+39.8%**; +34.3% is quoted instead,
which assumes that round was the anomaly - a judgement, not a measurement, so both are recorded. Every
other cell in the session has a spread across rounds under 1% except 27B tensor pp2048 (stock sd 2.96,
patched 3.36).

Four things to read out of it:

- **The pp512 numbers are patch 51, not the new ref.** Patch 51 landed on 2026-10-13, so the `v0.6.0`
  session below it never had it in the set - which is exactly why that table shows only +5.7% at pp512.
  51's own A/B at `v0.6.0` measured pp512 242.92 → 369.04 (**+51.9%**) and pp1024 **+29.1%** with decode
  flat, and this session reproduces that shape on both splits (+74.2% tensor, +47.1% layer, decode
  unchanged by it). It is the first full four-config session that has 49, 50 and 51 in it together.
- **Decode holds, within session drift.** +42.0% against +44.5% on 27B tensor, +36.1% against +37.6% on
  27B layer, +20.4% against +23.4% on 35B layer. Nothing here says the port lost decode.
- **Tensor-split prefill is now clearly positive on both models**: +14.9% (27B) and +14.2% (35B) at
  `-p 2048` / `-p 4096`, against +9.7% and +7.2% at `v0.6.0`. Layer-split prefill stays modest (+1.8%,
  +2.9%), as it has been all along.
- **The split recommendation is unchanged and now measured on this ref**: patched tensor beats patched
  layer on decode in absolute terms on both models (33.28 against 20.68, 77.60 against 66.41) and by
  more on prefill, so tensor is the arm to run.

An earlier 2-pass run at this ref, taken before this session, reported +14.0% pp2048 and +42.5% tg128 on
the 27B tensor cell. Superseded, but it is the run that found the `-DGGML_CUDA_NCCL=OFF` requirement:
with NCCL left at its default ON, both arms abort after load in `ggml_backend_cuda_comm_allreduce_nccl`
at `ncclGroupEnd()` on these cards, stock fails identically, and patch 49 never executes.
### Scripted - 2026-10-19, `v0.7.0`, 3x Tesla P100 - the third card, and why patch 49 stopped at two

A third P100 went into the box, and the first thing it showed is that the set's largest effect stopped at
two cards: `ggml_cuda_ar_pipeline_init` capped `n_devices` at 2 unless `GGML_CUDA_AR_NGPU` was set, so a
3-way tensor split silently lost patch 49. That cap has been widened by the end of this session - see the
flip below. Three arms, 5 interleaved rounds, same runner and same flags
as the 2-card session above, both arms built `-DGGML_CUDA_NCCL=OFF`, cooled to 45 C between configs. The
`NGPU=1` arm is the **same binary** as the default arm behind a wrapper that exports one variable, so
that pair isolates the gate from everything else. Rows in
`benchmarks/results/v0.7.0-p100x3-{tensor,layer}-{27b,35b}.csv`.

Qwen3.8-27B Q4_0 dense, `-sm tensor`:

| arm | pp512 | pp2048 | tg128 |
|---|---:|---:|---:|
| stock | 282.30 | 399.53 | 24.81 |
| patched, as shipped today | 364.79 | 408.31 | 28.79 |
| patched + `GGML_CUDA_AR_NGPU=1` | **503.11** | **561.80** | **38.95** |

Ornith-1.0-35B-A3B UD-Q4_K_M, `-sm tensor`:

| arm | pp4096 | tg128 |
|---|---:|---:|
| stock | 1018.26 | 41.03 |
| patched, as shipped today | 1042.76 | 55.70 |
| patched + `GGML_CUDA_AR_NGPU=1` | **1271.45** | **78.33** |

`-sm layer` at three cards, where the variable must make no difference:

| model / arm | pp512 | pp2048 | pp4096 | tg128 |
|---|---:|---:|---:|---:|
| 27B stock | 147.19 | 228.13 | - | 15.04 |
| 27B patched (either arm) | 218.9 | 234.1 | - | 20.4 |
| 35B stock | - | - | 1133.54 | 54.37 |
| 35B patched (either arm) | - | - | 1162.5 | 65.6 |

Read it in this order, because the obvious percentage is the least trustworthy one:

- **The gate on its own** (same binary, off to on): 27B tensor pp2048 **+37.6%**, decode **+35.3%**;
  35B tensor pp4096 **+21.9%**, decode **+40.6%**. Against stock the totals read +40.6% pp / +57.0% tg
  on the 27B and +24.9% pp / **+90.9%** tg on the MoE - but quote the within-build pair, because the
  MoE's *stock* arm falls when it gains a third peer (55.51 -> 41.03 decode from 2 to 3 cards), so that
  +90.9% is partly a weakened control.
- **The third card is real on the dense model, and nearly free on the MoE.** Patched with the gate on,
  2 cards to 3: 27B tensor 448.95 -> 561.80 at pp2048 and 33.28 -> 38.95 decode, while the 35B moves
  1257.15 -> 1271.45 and 77.60 -> 78.33, under 1%. On the MoE the gate buys protection against the
  fallback's cost, not extra throughput.
- **The layer control passes.** `NGPU=1` and the default agree within noise (1162.54 against 1162.55 at
  pp4096), which is what a layer split should do since it never reaches this code. The patch deltas on
  layer are also card-count independent: +2.9% / +20.4% at two cards, +2.6% / +20.6% at three.
- **Correctness was settled before anything here was believed.** At ctx 2048, ub 2048, three cards, the
  N-way reduce against the host fallback in the same build: PPL 2.8037 against 2.8025 (-0.04%), mean
  KLD 0.000343, 98.5% of positions keeping the same top token. It is not bit-identical - a different
  summation cannot be - and the gap is about 3x the 2-card equivalent at the mean and 12x at the max,
  which is why perplexity, not identity, is the bar for this one.
- **How it failed was the worse half: silently.** With the gate unset at three cards, nothing was
  logged - not by `llama-bench`, not by `llama-cli`, and the code does contain an init-failure warning
  on that path. A user adding a third card therefore lost ~38 points of prefill with no diagnostic. Why
  that warning never reaches the terminal is still open, and it matters more than the flip does.

**The flip (2026-10-19): 2..4 cards now take the internal AllReduce by default.** `GGML_CUDA_AR_NGPU=1`
lifts the ceiling to `GGML_CUDA_MAX_DEVICES` (16) for machines measured past four cards,
`GGML_CUDA_AR_NGPU=0` restores the original 2-device cap, and unset means four. The ceiling is
deliberately narrow rather than maximal: three cards are measured here and four are not, so a four-card
machine gets the internal path on code that has only ever run three peers - a stated risk instead of a
silent absence - and 5..16 stay behind the variable.

Before flipping, the shapes this session had not touched were exercised on three cards with the gate on.
pp8192 **529.02** against 390.88 with it off; tg512 **39.57** against 29.14; perplexity over 3 chunks at
ctx 4096 **3.7324** against 3.7314, 0.03% apart; and `llama-server -np 4` with six concurrent 32-token
completions - all six returned, and the log showed `graphs reused = 60` with `graph reallocs = 0`, so
patch 50's graphs and the N-way reduce coexist under multi-sequence serving. The only error-shaped line in
that log was upstream's own `llama_params_fit is not implemented for SPLIT_MODE_TENSOR` warning, present
in both arms.

After the rebuild the change measures inert exactly where it was already on: two cards give 451.82
pp2048 / 33.34 tg against the scripted 448.95 / 33.28. Three cards with nothing set in the environment
give **569.25 / 39.34**, and `GGML_CUDA_AR_NGPU=0` reproduces the pre-flip default's 409.21 / 28.95 - so
the escape hatch is the old behaviour, not something new.

### Scripted - 2026-10-12, `v0.6.0`, 45 patches - the same set before the port

The same four configurations, re-taken after the two defaults that changed this week: patch 49's
Pascal gate is now on, and patch 50 (`cuda-graphs-pascal`) is new and on. Same box, same runner, 5
rounds, arms interleaved with the order reversed on even rounds, both arms build line
`d81235049 (11429)` = `v0.6.0`, nothing exported in the environment for either arm. `-fa on -t 28
-ub 2048 -n 128`, `-p 512,2048` on the 27B and `-p 4096` on the 35B, cooled to 45 C between configs
(40-61, 44-57, 44-54 and 44-52 C in table order).

| Model | Split | pp t/s | | tg 128 t/s | | sd across rounds |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.8-27B Q4_0 dense | `-sm tensor` | 382.53 → **419.46** | **+9.7%** | 23.27 → **33.63** | **+44.5%** | 0.02 / 0.01 |
| Qwen3.8-27B Q4_0 dense | `-sm layer` | 221.28 → 223.60 | **+1.0%** | 15.27 → **21.01** | **+37.6%** | 0.00 / 0.01 |
| Ornith-1.0-35B-A3B UD-Q4_K_M | `-sm tensor` | 1091.65 → **1169.88** | **+7.2%** | 57.88 → **78.38** | **+35.4%** | 3.11 / 0.05 |
| Ornith-1.0-35B-A3B UD-Q4_K_M | `-sm layer` | 915.02 → **952.76** | **+4.1%** | 55.30 → **68.25** | **+23.4%** | 0.03 / 0.05 |

Read three things into that against the section below it, which is the same box one day earlier:

- **Prefill went from flat to positive on the tensor split** (27B `pp2048` −0.5% → +9.7%, 35B
  −0.7% → +7.2%) and the 27B also gained at pp512 (229.83 → 242.88, +5.7%). That is patch 49's gate
  being on by default, not a new effect — the AllReduce section below measured the same shift with
  the variable exported.
- **The tensor split is now faster on both models**, not just the dense one: 78.38 against 68.25
  decode on the MoE, 1169.88 against 952.76 prefill. The layer recommendation in the older table was
  an artefact of both arms using the host fallback.
- **Patch 50 costs the MoE layer arm about 3.5%** (70.64 → 68.25 decode across the two days, of
  which +12.5% went to tensor in the same build) and is worth +12.5% there. Measured against itself
  in one binary, not across these tables — see
  [`benchmarks/results/v0.6.0-p100x2-cuda-graphs.md`](benchmarks/results/v0.6.0-p100x2-cuda-graphs.md).

Rows are in `benchmarks/results/` as `v0.6.0-p100x2-{tensor,layer}-{27b,35b}-g50.csv` with the
`run.sh` table beside each. The softest number in the set is the 35B tensor stock arm: its decode
sd is 3.11 t/s, the highest of any arm here, so that +35.4% carries a couple of points of slack.
Perplexity on the shipped defaults is unchanged from the fallback paths it replaced (27B `-sm layer`
5.2538, 35B `-sm tensor` 4.8859, both to four decimals).

**2026-10-13, one more default.** Patch 51 asks cuBLAS for a blocked-accumulation algorithm on GP100
and changes nothing else about how this set runs. Measured inside one build (`GGML_CUDA_CUBLAS_ALGO_PASCAL=-1`
as the off arm, 5 interleaved rounds): 27B `-sm tensor` **pp512 +51.9%, pp1024 +29.1%**, pp2048 and
decode level, MoE at pp4096 within 0.6%. Perplexity over 81,920 tokens says it is also the more
accurate choice: 6.2785 with upstream's algorithm, 6.2658 with ALGO6, 6.2645 for all-fp32 matmuls —
ten times closer to fp32. Method and limits:
[`benchmarks/results/v0.6.0-p100x2-cublas-algo.md`](benchmarks/results/v0.6.0-p100x2-cublas-algo.md).

Two more quantizations of the same 27B were baselined the same day, and they make the staleness of the
table above explicit rather than theoretical: on **Q4_1** the patched arm is **+64.2% pp512, +8.3%
pp2048, +40.0% tg128** under `-sm tensor`, and on **UD-Q4_K_M** +58.9% / +7.9% / +41.9%. The pp512 gain
survives on `-sm layer` too (+41.4%, where no AllReduce runs at all), so it is patch 51 and the table's
prefill columns are stale at short prompts on every quantization — a full re-take is owed, not another
footnote. Decode also ranks by quantization there (34.58 / 33.63 / 28.79 t/s for Q4_1 / Q4_0 / Q4_K_M),
with Q4_K the slowest and the obvious target for the per-type matvec work still on the list. The file
also records the first MTP runs on this build, and they changed the plan: the built-in nextn head is
worth **+40.9%** at `--spec-draft-n-max 2` but the build's own default of 3 gives up 13% of that,
so `--spec-draft-n-max 2` is the flag to use. Porting the source fork's speculative-decoding work
turned out to be unnecessary — sampled drafts, the lossless rejection rule, `p_min` draft stopping
and the K/V catch-up are all already in `v0.6.0`, and the one thing missing (a fixed-width padded
verify) makes variable draft widths cheap on a build where the widths are already fixed. Measured,
not assumed: a probability stop here is neutral at 2-3 drafts and **−4.3%** at 5.
[`benchmarks/results/v0.6.0-p100x2-mtp-width.md`](benchmarks/results/v0.6.0-p100x2-mtp-width.md).
[`benchmarks/results/v0.6.0-p100x2-baseline-q41-q4km.md`](benchmarks/results/v0.6.0-p100x2-baseline-q41-q4km.md).

A third measurement that day closes a candidate instead of shipping one. The next item on the list was
the fork's tensor-parallel exchange overlap (`1f92a1b48`, +4.5% pp2048 on their board), so the
exchange was costed first: doubling the bytes on the wire (`GGML_CUDA_AR_BF16_THRESHOLD=0`) costs
**8.46% of pp2048**, which is any overlap's ceiling; doing the whole exchange with SM kernels instead
of the copy engines costs 0.96%, so the transport is not the problem; and the copy-engine chunk-size
heuristic is already at its optimum. The blocker is a CUDA primitive, not an amount of work — the
overlap needs the host-enqueued AllReduce to wait on an event recorded *inside* a captured prefill
subgraph, and on sm_60 under CUDA 12.4 an ordinary stream cannot: `cudaEventRecord` during capture
leaves no event node and the external wait is rejected outright, while a record node appended to the
graph is accepted and does not synchronise at all. **Not ported.**
[`benchmarks/results/v0.6.0-p100x2-exchange-cost.md`](benchmarks/results/v0.6.0-p100x2-exchange-cost.md),
probe source in [`benchmarks/tools/graph-event-wait.cu`](benchmarks/tools/graph-event-wait.cu).

That session then separated the two increments inside patch 49 for the first time and shipped the
cheap one. `GGML_CUDA_AR_H2D_STREAM` — stage 2's host-to-device copy on its own stream, so the second
copy engine overlaps the next device-to-host chunk — is worth **+3.47% pp2048 and +0.00% tg128** on
its own, measured as four interleaved arms on one binary. It is **on by default** now, `=0` opts out.
A throughput row cannot tell "faster" from "different-and-wrong" and this patch changes
synchronisation rather than arithmetic, so the gate was three comparisons of the stream arm against
the arm it replaces, same binary: paired perplexity identical to the last digit at both ubatches
(5.2281 / 5.2322), nine repeat runs — including an arm with the chunk size forced to its 256 KB floor
to multiply the cross-stream edges — all returning the same number, and a new **logit-level** check
([`benchmarks/logit-identity.sh`](benchmarks/logit-identity.sh)) scoring `--kl-divergence` of the
stream arm against the default arm's exported logits: **maximum KLD 0.000000**, bit-identical at every
position. The same test puts the *fallback* reduction at mean KLD 0.000093 / max 0.001835, so the
difference in summation order that does exist between reductions is now measured instead of argued.
Raw rows: [`…-ppl-46patch.txt`](benchmarks/results/v0.6.0-p100x2-ppl-46patch.txt),
[`…-logit-identity.txt`](benchmarks/results/v0.6.0-p100x2-logit-identity.txt),
[`…-h2d-default.csv`](benchmarks/results/v0.6.0-p100x2-h2d-default.csv).

The third candidate, `a8b274ea6` (hash the meta split-state cache, pool the per-depth source states,
`GGML_BACKEND_META_MAX_DEVICES` 16 -> 8), was built and measured on the card and is **not shipped**.
Timed from inside `ggml_backend_meta_graph_compute` on the 27B tensor split, the rebuild block it
optimises costs **14.0 ms** per shape change and the patch brings that to **6.2 ms** — their own
14-23 ms figure reproduces, and so does the halving. But the call that contains it averages 1374 ms
per ubatch, because the meta backend's `graph_compute` spans the GPU work it dispatches, so 8 ms is
0.6% of one ubatch and nothing at all in a throughput row, which llama-bench takes after warmup has
already paid the rebuild. Not worth a patch that resizes a struct in a public header. Diff kept at
`work-rebase/diag/C.lf.diff`, with the timing instrumentation beside it.

### Scripted - 2026-10-11, `v0.6.0`, 44 patches, internal AllReduce off in both arms

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
  spread on each), so the older figure was flattered by a slow control. Quote +5.7% **for that
  build** — with the gate now on by default the same config reads +35.4%, in the shipped table above.

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
- **None of it was, at the time, a correctness result.** With the gate on, the reduction genuinely is
  a different one than the fallback's, and when these rows were taken there was no perplexity and no
  logit-level number beside them. Patch 50's history is the warning that made that unacceptable —
  upstream's first attempt at exactly this stream split measured "just as fast" and returned
  perplexity **7.68 instead of 3.53**. Both checks now exist for this patch:
  [`benchmarks/results/v0.6.0-p100x2-tensor-27b-perplexity.md`](benchmarks/results/v0.6.0-p100x2-tensor-27b-perplexity.md)
  and, since 2026-10-13,
  [`benchmarks/results/v0.6.0-p100x2-logit-identity.txt`](benchmarks/results/v0.6.0-p100x2-logit-identity.txt)
  — the fallback reduction sits at mean KLD 0.000093 / maximum 0.001835 against the internal path.

The patched arm ran warmer than stock inside these sessions (52-62 C against 44-62 C), so the
gains are if anything understated. Provenance per row in
[`benchmarks/results/v0.6.0-p100x2-tensor-27b-ar49on.csv`](benchmarks/results/v0.6.0-p100x2-tensor-27b-ar49on.csv)
and
[`…-tensor-35b-ar49on.csv`](benchmarks/results/v0.6.0-p100x2-tensor-35b-ar49on.csv). The two
`…-aroff` files beside them are the same two tensor configs re-taken after patches 49, 50 and the
N-GPU port were folded into one, which is how the merge was shown to be free at two cards: 30.14 →
30.07 and 70.64 → 70.66 decode, with both arms of the 27B drifting the same way.

**The Pascal gate is on by default**, so a pre-Volta tensor split gets that table without being
asked; since 2026-10-13 the H2D stream is on by default too, and since 2026-10-19 the reduce covers
2..4 devices with nothing set, so `GGML_CUDA_AR_NGPU` only moves that ceiling now - `=1` raises it to
`GGML_CUDA_MAX_DEVICES` (16), `=0` puts it back at two. The gate is the one part of
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

**Read the vintage before you quote any of this.** Only the first section above is this set: everything
from the `v0.6.0` heading onward is the 46-file `v0.6.0` tree, produced with five patches (08, 34, 39,
46, 48) that this branch no longer contains, because upstream absorbed or obsoleted them
([docs/patches.md](docs/patches.md)), and regenerated against a different llama.cpp. They stay because
they are the control the `v0.7.0` session is read against, not because they describe what ships.
Exactly one output-identity check exists beside any of it - patch 49's H2D stream at `v0.6.0`, maximum
KLD 0.000000 - and there is none at this ref.

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
patches/            72 .patch files, applied in numeric order (52-83 are the P60 block)
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
- The **P60 block** (52-83) is written for one model family - Qwen3.5/3.6 MoE at 2048 hidden,
  Q4_K/Q5_K experts - and is `sm_60`-gated as a whole. On the dense 27B it measured **−1.8% pp512**
  with both switches armed and flat decode; `GGML_CUDA_P60_MEGA=0` recovers the difference, which is
  another reason that switch is not on by default. Its persistent kernels crash under
  `--spec-type draft-mtp` and its persistent GDN half does not match llama.cpp v0.8.0's graph - both
  are in [What to expect](#what-to-expect) with the numbers.
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
  in the code — and so since 2026-10-13 is the H2D stream, which puts the copy-engine return leg on a
  second stream (`GGML_CUDA_AR_H2D_STREAM=0` puts it back). `GGML_CUDA_AR_NGPU` no longer decides whether
  the reduce runs at all: since 2026-10-19 the internal AllReduce covers 2..4 devices by default, `=1`
  lifts that to `GGML_CUDA_MAX_DEVICES` (16), and `=0` restores the old 2-device cap. Before that change a
  third card silently lost the whole of patch 49 — nothing logged (see the 3x P100 section above).
  `GGML_CUDA_AR_PASCAL=0` restores upstream's behavior exactly. The gate changes which reduction runs
  across a tensor split, so it can change output; it was flipped on the strength of a paired
  perplexity check whose widest spread is 0.46% and, now, a logit-level KLD against the fallback
  (see [What to expect](#what-to-expect)), and the N-GPU half was validated by its author on *virtual*
  devices; as of 2026-10-19 it has run on three real P100s - worth +37.6% pp2048 and +35.3% decode over
  the same binary with the gate off - but never on four. It compiles with nvcc 12.4 at `sm_60`, and on two real cards
  the two parts measure **+13.6% pp / +45.5% decode on the dense 27B and +9.9% / +21.4% on the 35B MoE**
  against the same arms on the fallback — better than qr0n's +10.1% / +9.6% at `b10660` — with the
  stream since separated out of that: **+3.47% pp2048, +0.00% tg128** on its own, bit-identical
  logits. One part of it is **not** behind any variable: making the
  reduction N-way rewrites both kernels, and that rewrite runs at two cards too, where it is meant to
  be arithmetic-identical (`w0 + w1` is what `own + peer` already was). It was re-taken to prove that:
  30.14 → 30.07 and 70.64 → 70.66 decode against the pre-merge build.
- Patch **51 `cublas-algo-prefill-pascal` is ported too**, from
  [Kmic-68/llama.cpp](https://github.com/Kmic-68/llama.cpp/tree/p100-optimizations) (an
  independently developed P100 fork, MIT), not from qr0n. It is on by default and reversible with
  `GGML_CUDA_CUBLAS_ALGO_PASCAL=-1`. That fork is the source of the next few candidates, and its
  numbers are its own model and quantization until we re-measure them here.
- Nothing in this repo compiles the CUDA sources. Applying cleanly is not the same as building, and
  building is not the same as measuring. The `v0.6.0` patched tree *has* been built and run on a P100
  box - that is what every scripted row above is - but `test-backend-ops` has never been run against
  it, and no `-npl` or perplexity number exists beside any of these gains. The CUDA patches are
  therefore verified by throughput and by eye, not by an op-level correctness suite. The `v0.7.0` arms
  build at `sm_60` as well and have had the one quick run above - neither has been near
  `test-backend-ops`.
