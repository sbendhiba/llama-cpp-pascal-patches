#!/usr/bin/env bash
# Identifier sweep for a rebased patch set: every environment knob the set introduces has to still be
# read by the patched tree, and every knob the docs present as a switch has to exist in the code.
#
#   ./benchmarks/knob-sweep.sh [patched-tree] [repo]
#
# Why this is a gate and not a nicety: a knob whose read disappeared in a rebase does not fail the
# build, does not fail `patch -F0`, and does not fail any measurement - it just stops being a switch.
# The docs keep advertising `NAME=0`, the user sets it, and nothing happens.
#
# The set reads its knobs through two forms, `getenv("NAME")` and the p60 helper `p60_env_on("NAME")`,
# so both are scanned; matching only the first reports GGML_CUDA_P60_FUSE_NORM as missing when patch 54
# converted it to the helper form.
#
# The docs direction is intersected with the patch names on purpose: prose backticks MUL_MAT and
# RMS_NORM and CUBLAS_GEMM_ALGO6, and none of those are switches. What is left after the intersection
# is a name the docs treat as a knob, some patch introduced, and nothing reads - a dead switch.
#
# A name legitimately disappearing is fine and says so: patch 86 deleted GGML_CUDA_P60_MEGA_GRAPHS
# along with 85's capture refusal. Read the docs entry before "fixing" a gap this sweep reports.
set -u
ARM=${1:-/root/llama.cpp/b1b770d-patched}
REPO=${2:-/root/llama-cpp-pascal-patches}

extract() {
    grep -rhoE '(getenv|p60_env_on) *\( *"[A-Z0-9_]+" *\)' "$@" 2>/dev/null \
        | sed -E 's/.*"([A-Z0-9_]+)".*/\1/' | LC_ALL=C sort -u
}

extract "$REPO"/patches/*.patch > /tmp/knob-sweep-patch.txt
( cd "$ARM" && extract ggml/src common src tools tests ) > /tmp/knob-sweep-tree.txt

echo "patched tree : $ARM"
echo "repo         : $REPO"
echo "knobs introduced by the set   : $(wc -l < /tmp/knob-sweep-patch.txt)"
echo "knobs read by the patched tree: $(wc -l < /tmp/knob-sweep-tree.txt)"

echo "--- introduced by a patch, not read anywhere in the patched tree ---"
LC_ALL=C comm -23 /tmp/knob-sweep-patch.txt /tmp/knob-sweep-tree.txt | sed 's/^/  /'

# `NAME` or `NAME=0` inside backticks, and a numeric value can be signed - `GGML_CUDA_CUBLAS_ALGO_PASCAL=-1`
# is a row in the README table and a pattern without the [-+] skips it, which is a silent false pass
grep -rhoE '`[A-Z][A-Z0-9_]{5,}(=[-+0-9A-Za-z]*)?`' "$REPO"/README.md "$REPO"/docs/*.md 2>/dev/null \
    | sed -E 's/^`//; s/`$//; s/=[-+0-9A-Za-z]*$//' \
    | LC_ALL=C sort -u | LC_ALL=C comm -12 - /tmp/knob-sweep-patch.txt > /tmp/knob-sweep-doc.txt
echo "--- named in the docs as a switch, not read by the patched tree ---"
LC_ALL=C comm -23 /tmp/knob-sweep-doc.txt /tmp/knob-sweep-tree.txt | sed 's/^/  /'
