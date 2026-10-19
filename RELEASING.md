# Rebasing onto a new llama.cpp

The patch set is generated against one specific llama.cpp tag, recorded in
[README.md](README.md). Upstream moves daily; the patches only apply at zero fuzz and zero offset on
that tag. This is the procedure for moving them, and the failure modes that make it more than a
mechanical re-apply.

Everything below is done on a **clean tree at the new tag**, never in a working copy you also build
from. Worktrees keep it tidy:

```sh
git -C llama.cpp worktree add ../rebase v0.7.0      # pristine tree at the new tag
```

## 1. Find out what actually broke

Apply the current set in numeric order and collect every failure:

```sh
cd ../rebase
for p in ../llama-cpp-pascal-patches/patches/*.patch; do
  b=$(basename "$p")
  patch -p1 -F0 --batch --no-backup-if-mismatch < "$p" >/dev/null 2>&1 || echo "REJECT $b"
done
```

Two things this list gets wrong, both of which cost real time if you believe it:

- **Cascade failures.** A patch whose prerequisite was skipped also rejects. On the `v0.5.0` to
  `v0.6.0` rebase 16 patches rejected and 3 of them (24, 25, 27) were pure cascades: they applied
  with no changes at all once their prerequisite was ported.
- **Patches that "applied" but shouldn't be trusted.** Anything reporting an offset needs
  regenerating too, because the whole zero-offset contract is what makes the next rebase diagnosable.

So treat this as a work queue, not a verdict. Work it **in application order**: later patches read
lines earlier ones wrote, and a fix late in the sequence can invalidate an earlier one.

## 2. For each failure, decide which of three it is

**Fixed upstream — delete it.** Upstream implemented the same thing, or made it unnecessary. Delete
the `.patch`, remove its row from the README table and its entry from `docs/patches.md`, and write
down what replaced it, with the upstream commit or PR number. This repo has shrunk four times on
this rule: 43 went at `v0.5.0` (upstream added `DSV4_HC_PRE`/`DSV4_HC_POST`), 44 went at `v0.6.0`
(upstream added the indexer key pool), 48 lost most of its reason to exist at `v0.6.0`
(upstream fuses one `RMS_NORM`->`SCALE` pair itself), 34/39/46 went at `b11539` (upstream's
`llama_moe_cache`, GDN lanes-per-column), and 08 went at `v0.7.0` (upstream gives `sm_60` the whole
MMVQ `MUL_MAT_ID` ceiling it existed to raise). Carrying a patch upstream already provides is
not conservative - it allocates a second copy of a feature and can silently fight the real one.

**Moved — port it and regenerate.** The intent is still needed, the code around it shifted.

**Replaced — usually drop it.** Sometimes the surface the patch hooked into no longer exists, and
porting it is a redesign plus a fresh measurement, not a rebase. That is how 45 went at `v0.6.0`:
`set_input_qsa` became `set_input_kpool`, its tensors were replaced, and the scoring it fed moved
into a fused node. Say so explicitly in the docs, and say what remains unmeasured.

Before you port anything, check whether upstream **retuned a constant** your patch overrides.
Several patches here exist only because of a value upstream tuned for other hardware. If it applies
and the value is now the same, it does nothing. If the value changed for the better, delete the
patch and note why.

Worse, and this is the case `v0.7.0` caught: the patch can edit a value the caller **no longer
reaches**. Upstream put an early `return` for `cc == GGML_CUDA_CC_PASCAL` above the table that
08 rewrites, on both the host dispatch and the device constexpr, so 08 applied at zero fuzz, zero
offset, with byte-exact context - and did nothing at all, because no path to it survives on this
card. All three gates pass a patch like that. The only check is to read the caller: for every value
a patch sets, find the branch that decides whether the card takes it, and confirm that branch still
leads to the patched code. Note that a patch like this can sit in a published branch for a whole ref
before anything notices - `1623d8ce4` is an ancestor of `b11601`, so 08 was already dead there.

## 3. Regenerate every patch at exact context

A patch whose context shifted must be re-cut, or the next rebase cannot tell "moved" from "broken":

```sh
base=$(git rev-parse HEAD)                 # tree state below this patch
<apply or hand-port the patch>
git add -A && git commit -m "rebase: <name>"
git diff "$base" HEAD > patches/<name>     # exact context, by construction
```

Keep one commit per patch. If you fix a placement mistake after the fact, fold it into that patch's
own diff (`git diff <base> HEAD -- <the files that patch owns>`) rather than letting the correction
live only in a later commit - the file on disk and the patch that produces it must not diverge.

Folding a change into a patch in the middle of the set means resetting to that patch's **parent** and
re-running the loop from there, so the patches after it get rebuilt too. `git reset --hard <sha>^^` is a
grandparent, not a parent: doing that for patch 49 dropped patch 47's commit, and the only reason it was
caught is that gate 2 then reported a diff in `ggml-backend.cpp` - a file the fold never touched. A
mystery file in the identity gate is a history error, not a patch error.

## 4. Gates to run before touching a card

**Clean-room apply.** Fresh tree at the new tag, apply every regenerated patch, and require no
`FAILED` and no `offset` anywhere.

**Tree identity.** The tree produced by the clean-room apply must be **byte-identical** to the tree
you reviewed patch by patch. This is what proves the patch files contain exactly what you examined,
and nothing accumulated in between.

**Host-side syntax check.** Compile every non-CUDA translation unit the set touches:

```sh
g++ -fsyntax-only -std=c++17 -D_GNU_SOURCE \
  -Icommon -Iinclude -Iggml/include -Iggml/src -Iggml/src/ggml-cpu -Isrc \
  common/sampling.cpp ggml/src/ggml-backend.cpp src/llama.cpp src/llama-context.cpp \
  src/llama-model.cpp src/llama-sampler.cpp src/models/qwen4exp.cpp tests/test-backend-ops.cpp
```

Do not skip this because "the patches applied". A hunk that matches identical context in two places
lands in the first one, and `-F0` is perfectly content with that: on `v0.6.0` a `models.h` edit
intended for `llama_model_qwen4exp::graph` landed in `llama_model_qwen35::graph`, which upstream had
grown in the meantime. Only the compiler noticed. Same class of problem: a patch that adds a
parameter to a function gains **new callers** upstream - `graph_mtp` was added to qwen4exp in the
same window and had to be updated by hand, and a helper that only one graph constructor called went
uncalled by the second one.

**Identifier sweep.** Grep the patches for every `llama_*` / `ggml_*` / `GGML_*` name they use and
confirm each still resolves in the patched tree. Renames produce exactly this failure: applies
cleanly, references something that no longer exists.

Then, on the hardware: `test-backend-ops` full run, and `llama-bench` with both arms per
[benchmarks/README.md](benchmarks/README.md).

## Dropping a single patch

Dropping one from the middle generally means rebasing the rest, so find the real dependency set
instead of guessing: start from the working prefix and remove patches one at a time, keeping only
removals after which the target still applies at zero fuzz and zero offset. For `12-mmvq-f16-sm60`
on `v0.6.0` that came out as 02, 07, 08, 09, 10 - and 01, 04, 05, 11 were genuinely unnecessary,
which no one could have guessed from the file list.

Expect the dropped set to apply at small offsets: those are line-number shifts with exact context,
not fuzz, and they are safe - but say so, because the zero-offset claim in the README is written for
the complete set.

## Also update

- **re-point this repo's own tag at `main`** — `git tag -f v0.6.0 main && git push -f origin refs/tags/v0.6.0`.
  That tag is what the README's apply block checks out, so anything committed after it is invisible to
  anyone following those instructions. On 2026-10-11 it was stale twice over: first by a patch set
  whose count the README had already changed, then by a default value. It is a published ref, so
  moving it is a force-push and the one action here that needs a human to decide the moment is right
- the tag in `README.md`, in `docs/patches.md`'s header, and in `benchmarks/results/` file names
- patch counts, **recomputed** from the table rather than edited by hand: the scope counts, the
  off-by-default counts, and the "N of the M are off by default" sentences are all wrong the moment a
  patch is deleted
- every cross-reference to a deleted patch (search for its number - other patches' entries cite it
  as justification, and those justifications are now stale)
- `docs/patches.md`: add a dated deletion note next to the existing ones, saying which upstream
  change made it redundant
