# Benchmarks

This directory is where the set gets re-measured, once per llama.cpp release, on the hardware you
have. It exists because the README numbers were taken on `b10133` and `v0.4.0` and had never been
re-taken against the tree that ships. That has changed for `v0.6.0`: four scripted 5-round runs on
2x P100 - `{tensor,layer}-27b` and `{tensor,layer}-35b`, both arms `d81235049`, `-ub 2048` - are in
`results/` as one row per arm, round and test with the temperature it was taken at, and they moved
one hand-run number (see the cross-check in the [top-level README](../README.md)).
[`results/v0.6.0-p100x2-manual.md`](results/v0.6.0-p100x2-manual.md) keeps the hand-run provenance
that predates the script. A second hand-run set, on a qwen4exp MoE too large to fit the cards and
the only place this set loses prompt processing, is in
[`results/v0.6.0-p100x2-qwen4exp-manual.md`](results/v0.6.0-p100x2-qwen4exp-manual.md).

A follow-up session the same day added four more scripted runs: `…-tensor-27b-aroff` and
`…-layer-35b-aroff` re-take two of those configs after patches 49, 50 and the N-GPU port were folded
into one file (they reproduce them, which is what makes the merge safe), and `…-tensor-27b-ar49on`
and `…-tensor-35b-ar49on` take the same configs with `GGML_CUDA_AR_PASCAL=1
GGML_CUDA_AR_H2D_STREAM=1` exported for the session - the first measurement of those variables on
this tag. The `env` column in each csv is the proof of which is which; note that a comparison
*between* those files is a cross-session comparison, while the arms *inside* one file are
interleaved. Those two files are now historical: the gate became the build default afterwards, so a
current tensor-split run needs no environment at all.

[`ppl.sh`](ppl.sh) is the other half of the protocol. `run.sh` answers "is it faster", which cannot
distinguish a fast reduction from a different-and-wrong one, and patch 49 is exactly that case. It
scores a real corpus through one binary in four arms - internal, `GGML_CUDA_AR_PASCAL=0` fallback,
the second H2D stream, and `-sm layer` which does no cross-device reduce - at two ubatch sizes, and
reports whether each arm actually reached the internal path. Its output is
[`results/v0.6.0-p100x2-tensor-27b-perplexity.md`](results/v0.6.0-p100x2-tensor-27b-perplexity.md),
which is also the evidence behind that flip. Perplexity is deterministic, so unlike `run.sh` it needs
neither interleaving nor a cooling wait.

[`results/v0.6.0-p100x2-cuda-graphs.md`](results/v0.6.0-p100x2-cuda-graphs.md) is a **rejected**
candidate rather than a measurement of this set: it lowers the CUDA-graph architecture gate to sm_60
(the one-line idea of llama.cpp#27721, one arch further than that PR goes) and finds 27B decode and
35B-layer prefill 0.8-3% *slower* for one tensor-split MoE case 12% faster. It is kept because the
negative is the useful part, and because it records how to A/B a change that has no environment
switch: one build, two arms, the off arm reached through a wrapper that exports the kill variable.

## Run it

```sh
./run.sh --tag v0.6.0 --gpu p100x2 --reps 5 \
    --arm stock=/opt/llama-v060/build/bin \
    --arm patched=/root/llama.cpp/build/bin \
    -- -m /root/models/Qwen3.8-27B-Q4_0.gguf -fa on -sm tensor -t 28 -p 512,2048 -n 128
```

Everything after `--` is passed to `llama-bench` untouched; the script adds `-r 1 -o jsonl` per call
so each round is one sample and the interleaving is genuine. It alternates the arm order on even
rounds, records the GPU temperature beside every row, and appends to `results/<tag>-<gpu>.csv`. The
summary it prints reports the mean, the spread **across rounds**, the round count and the temperature
range per arm, and only covers the run you just did even though the csv keeps accumulating.

Two details worth knowing before you change the parser. A llama-bench `jsonl` record has **no test
name** - `test::get_fields()` ends at `stddev_ts`, and the `pp2048` / `tg128` label only exists in
the md and csv printers, built from `n_prompt`, `n_gen` and `n_depth`. So `run.sh` rebuilds it the
same way. And `stddev_ts` from a `-r 1` call is a single sample, which is why the summary recomputes
it. Both of these were found by losing a run to them: the first version of this script filtered
records on `"test"`, matched nothing, and reported an empty table only after all five rounds had
already run. It now stops on the round that produced no records.

Two details that cost a wasted run to find out. A `jsonl` record carries **no test name** - the
`pp2048` / `tg128` label exists only in the md and csv printers, where llama-bench builds it from
`n_prompt`, `n_gen` and `n_depth` - so the script rebuilds it the same way. And its `stddev_ts` is
meaningless at `-r 1`, so the summary recomputes the spread across rounds instead of quoting it. A
call that returns no records stops the run there, rather than finishing cold with an empty file.

An arm is a build directory, not a build flag. Both arms must be **the same llama.cpp tag**: build
the unpatched one from the pristine tree and the patched one from the tree after the apply loop, and
put them in separate prefixes so neither `build/` contaminates the other. Comparing a patched tree
against a different upstream version measures the version, not the patches - that mistake produces
confident, useless numbers.

To measure one patch's increment rather than the whole set, make the second arm the same binary with
that patch's environment variable set. For example `--arm all=/root/llama.cpp/build/bin --arm no22=/root/llama.cpp/build/bin`
plus `LLAMA_DEC_SLOTS=0` in the shell for the second arm's run - run.sh records the variable names in
the `env` column, so the arms stay distinguishable.

## What to measure

Two models minimum, one of each class the set was tuned for, plus whatever you actually run:

| | purpose |
|---|---|
| a dense model, all layers on GPU | the set should be roughly neutral here; a regression shows up first |
| a MoE with routed experts, MTP speculation on | the decode path the patches were written against |
| long context with a quantized KV cache and `-fa on` | exercises the flash-attention and compute-buffer patches |

`-p 512 -n 128` and `-p 2048 -n 128` are the smallest pair worth recording. Use `llama-bench -o md`
for a quick look, this script for anything you intend to keep.

## Naming

```
results/<llama-tag>-<gpu>x<count>.csv                  every round, every arm, one row each
results/<llama-tag>-<gpu>x<count>-summary.md           the table run.sh printed
results/<llama-tag>-<gpu>x<count>-manual.md            provenance for numbers taken by hand
```

`p100x1`, `p100x2`, `k620x1`, `rx580x1` - the GPU is part of the identity of a number here, because
these patches are gated on architecture features and the same patch can be +23% on one card and
noise on another.

When one release gets several configurations, put the configuration in the `--gpu` field and keep the
results in one directory per run, which is how the `v0.6.0` session was organised: `--gpu
p100x2-layer-35b` writes `v0.6.0-p100x2-layer-35b.csv`, whose rows also carry the full `bench_args`,
so a file name is never the only record of what produced a number.

`run.sh` writes the first two. The third is for measurements taken straight from `llama-bench` before
the script existed, or for configurations the script has not re-taken yet: it carries the per-cell σ,
the arm definitions and an explicit list of what was not recorded, so a hand-run number can be quoted
with its weaknesses attached. `v0.6.0-p100x2-manual.md` is that file, and the scripted csv next to it
now covers the 27B tensor-split case it used to describe alone.

## Before a number goes in the README

- both arms on the same tag, same tree, differing only by the patches
- rounds interleaved with the order reversed (run.sh does this)
- the card cooled to a stated temperature before each arm - run.sh records what it actually was, and
  a round taken 15 C above your threshold is a round to discard, not to average in. Between runs in a
  session, wait for it: the `v0.6.0` session waited back down to 45 C before each of its four runs and
  logged the wait, so the ranges in the README start cool and stay within about 12 C
- the standard deviation and the round count shown next to the mean
- the arm definitions and any environment variables, in the row
- every flag that changes the comparison in the row or the filename, `-ub` above all: the same 27B
  tensor case measured 225.00 t/s at llama-bench's default `-ub 512` and 380.06 at `-ub 2048`, while
  its decode gain moved less than half a point
- for anything that changes output, `llama-bench -npl 5` or a perplexity run with `-ub 1` beside it,
  because throughput alone cannot tell a win from a wrong answer

[../docs/benchmarking.md](../docs/benchmarking.md) is why each of those rules exists, including the
two cases where a well-built microbenchmark predicted an effect that did not exist in the model.
