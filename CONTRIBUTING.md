# Contributing

Issues and PRs welcome. The most useful contribution is a **measurement on hardware this repo does
not have** - P40, GTX 10xx, Maxwell, or anything Turing and later - because every non-`sm_60` scope
tag here is an inference from a single GP100. Second most useful: a patch that deletes other patches,
with the upstream commit that made them redundant.

It is a private patch set, not a fork: no branch of llama.cpp to merge, and nothing here is
submitted upstream. The value is in the measurements as much as in the code, and the repo is expected
to shrink as upstream fixes things - see [RELEASING.md](RELEASING.md).

## Adding a patch

1. **One concern per patch.** If it fixes two things, write two patches, because the next rebase has
   to classify each one separately as *deleted upstream*, *moved*, or *still needed*.
2. Take the next free number. Numbers are never reused, so the gaps (03, 43, 44, 45 here) are the
   history; `docs/patches.md` explains each one.
3. Generate it from a commit-per-patch sequence against the current tag with `git diff`, so it lands
   at **zero fuzz and zero offset**. A patch that needs an offset today needs a wrong place tomorrow.
4. Put the evidence **in the patch itself**, as comments next to the code it justifies: what was
   measured, on what, the effect size, and the alternatives that were tried and rejected. The entry
   in `docs/patches.md` is an index, not the argument.
5. Add a row to the README table and a section to `docs/patches.md`, with a scope tag:
   `sm_60`, `pre-Volta`, `pre-Turing`, `no DP4A`, `CUDA`, `CUDA (delta-net)`, `host`, `model`,
   `all archs`.

## Rules a patch has to satisfy

**Output.** Unless the patch says otherwise, output must be **bit-identical** to the unpatched
build, and the entry must say how that was shown. If it does change output - a different reduction
order, a fused epilogue, a kernel that changes launch geometry - then: state exactly when it changes,
measure the difference (perplexity, `llama-bench -npl`, top-k log-prob comparison), and default it
**off** behind an environment variable with a documented kill switch. Several patches here are on by
default and each one says so with its evidence; the rest are opt-in.

**Scope.** Architecture-specific gating needs an architecture-specific *reason*. `sm_60` gating is
for the GP100 DP4A/HFMA2 asymmetry; a scheduler or sampler fix has no business being gated. Changing
an unconditional constant that every GPU sees (patch 07 does) is the hardest thing to justify here,
because the tuning behind the current value was done on hardware that can be measured by CI and yours
cannot.

**Memory.** A patch that grows a buffer or adds a scheduler needs a VRAM number. On a 16 GiB card
the difference between +0.38 points of throughput and the model not loading is not a tradeoff.

**Tests.** A new or changed `ggml` op needs `test-backend-ops` cases, and `test-backend-ops` must be
run on the card with the full set applied, not with the patch alone - patches interact through the
compute-buffer layout, which is how fusion decisions get made. Report the pass count and anything
that failed intermittently, and whether the unpatched build fails the same case.

## Measurement protocol

Anything quoted in the README or in `docs/patches.md` has to survive this, and
[docs/benchmarking.md](docs/benchmarking.md) explains why each rule exists:

- arms **interleaved**, order reversed for the second half, GPU cooled to a fixed temperature before
  each arm. A P100's boost sags from 1328 to 1101 MHz over a long benchmark, which is larger than
  most of the effects being measured.
- report the standard deviation and the number of rounds; if the effect is inside the spread, say so
- never conclude from a standalone microbenchmark or from `nvprof` token spans - both have
  mispredicted here. Per-kernel durations in situ are trustworthy; resolve effects below the noise
  floor with paired A/B, not by running something longer
- perplexity is measured with `-ub 1` (and `-ub 5` for anything touching speculative verify batches)
  - plain `llama-perplexity` runs batched prefill and never touches the decode kernels most of these
  patches change

## Reporting a measurement on your hardware

Use [benchmarks/run.sh](benchmarks/run.sh); it records the build, the driver, the GPU count and the
environment variables alongside the numbers, which is the difference between a data point and an
anecdote. Put the output in `benchmarks/results/` named `<llama-tag>-<gpu>-<n>x<gpu>.<ext>`, and
include both arms (patched and unpatched) from the same tree.

## Code style

The patches land in llama.cpp, so they follow llama.cpp's conventions: 4 spaces, no trailing
whitespace, `void * ptr`, `int & a`, basic loops, minimal templates. Comments stay 1-2 lines unless
they are carrying a measurement. No unicode arrows or em-dashes in code comments - use `->` and `-`
(they are fine in `docs/`, which is not compiled). Prefer extending an existing table or predicate
over adding a new file; a new `.cu` is a maintenance ask, not a formatting detail.

## Submitting upstream

Don't, on this repo's behalf, and not with AI-written descriptions. llama.cpp's `AGENTS.md` and
contributing guide prohibit automated submissions and AI-authored PR text, and the penalty is a
contributor ban. If a patch here is worth upstreaming, open an issue first to gauge interest, write
it yourself, and be ready to defend every line - a merged patch is somebody's responsibility forever.
