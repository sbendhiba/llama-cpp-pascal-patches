# llama-p100-patches

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

From the top of a clean llama.cpp tree at `b11539`:

```sh
git clone --branch b11539 https://github.com/ggml-org/llama.cpp
cd llama.cpp
for p in ../llama-p100-patches/patches/*.patch; do
  patch -p1 -F0 < "$p" || { echo "FAILED: $p"; break; }
done
```

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

Whole-set against stock on one P100, end-to-end decode from `llama-server`, geometric mean over
three prompts, MTP speculation and a realistic sampler:

| Model | Stock | Patched | |
|---|---:|---:|---:|
| Qwen3.5 9B dense (Q4_1, MTP head Q4_1) | 76.99 t/s | **136.11 t/s** | **+76.8%** |
| Qwen3.6 35B-A3B MoE (Q2_K_XL, dense layers Q5_1) | 67.01 t/s | **120.57 t/s** | **+79.9%** |
| Qwen3.8-Flash-Next MoE (UD-IQ3_XXS, q4_0 KV, ctx 262,144, 40k deep) | 8.63 t/s | **20.47 t/s** | **+137%** |

**Read the vintage before you quote these.** Rows 1 and 2 were taken on `b10133`, row 3 on
`v0.4.0`, and none has been re-taken against the tree that now ships. Each rebase since has been
measured against the previous patched tree rather than against stock, so these are several unmeasured
rebases away from what actually runs today. The v0.4.0-to-v0.5.0 delta was +3.4% and +1.5% on
prompt processing, flat to +1.0% on decode; the `v0.6.0` rebase was not measured on a card at all,
and neither was this `b11539` one - which also deletes four patches (34, 39, 46, 48, explained in
[docs/patches.md](docs/patches.md)) that those tables partly describe. The one `b11539` figure that
does exist is decode **+29.1%** for a dense 27B Q4_0 across two P100s with `-sm tensor`, but its
control arm was `master`, not `b11539`, so it is a data point and not a result. Full provenance in
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
- Everything was measured on **one** card. Patches scoped to `sm_60`, `pre-Volta` or `pre-Turing`
  change values upstream tuned for newer hardware, and `pre-Turing` includes Volta, which was never
  measured. Patch 07 is not gated at all. Measure before using any of them on Turing and later.
- `patch 22 decode-sched-slots` conflicts with the meta backend and **disables itself** under
  `-sm tensor`; multi-GPU otherwise runs the whole set, but no multi-GPU number here is measured.
- `CUDA (delta-net)` patches only fire on models with a gated delta-net block (Qwen3-Next, Qwen3.5).
  Elsewhere they never fire, and the only cost is the per-graph pattern match.
- Nothing in this repo compiles the CUDA sources. Applying cleanly is not the same as building, and
  building is not the same as measuring.
