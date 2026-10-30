#!/usr/bin/env bash
# Pull llama.cpp at the ref this branch is maintained against and apply the patch set to it.
#
#   ./scripts/patch-llama.sh [target-dir] [options]
#
# Options
#   --ref REF          llama.cpp tag or branch to check out        (default: the PINNED_REF below)
#   --commit SHA       require HEAD to be this commit after checkout (default: the PINNED_COMMIT below)
#   --source URL       llama.cpp remote to clone                   (default: ggml-org/llama.cpp)
#   --check            apply the whole stack to a throwaway copy and report, leaving the target pristine
#   --build            configure and build after applying          (CUDA, sm_60, NCCL off)
#   --arch NUM         CUDA architecture for --build                (default: 60)
#   --jobs N           build parallelism                            (default: nproc)
#   --force            re-apply onto an existing target directory
#
# Patches are applied in numeric order with `patch -p1 -F0`: zero fuzz, zero offset, stop at the first
# failure. That is the whole point of this script - a fuzzy apply can land a hunk in the wrong place and
# still exit 0. If the ref check fails, the patches are not applied at all.
set -uo pipefail

PINNED_REF="v0.8.0"
PINNED_COMMIT="8642541b2"
SOURCE="https://github.com/ggml-org/llama.cpp"

here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
target=""
ref="$PINNED_REF"
commit="$PINNED_COMMIT"
check=0 build=0 force=0 arch=60 jobs=$( (nproc 2>/dev/null || echo 4) )

while [ $# -gt 0 ]; do
    case "$1" in
        --ref) ref="$2"; shift 2;;
        --commit) commit="$2"; shift 2;;
        --source) SOURCE="$2"; shift 2;;
        --check) check=1; shift;;
        --build) build=1; shift;;
        --arch) arch="$2"; shift 2;;
        --jobs) jobs="$2"; shift 2;;
        --force) force=1; shift;;
        -h|--help) sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 0;;
        *) target="$1"; shift;;
    esac
done
[ -n "$target" ] || target="$here/../llama.cpp-$ref-patched"

patches=$(ls "$here"/patches/*.patch 2>/dev/null | wc -l)
[ "$patches" -gt 0 ] || { echo "no .patch files in $here/patches" >&2; exit 2; }

echo "patch set : $here ($patches patch files)"
echo "llama.cpp : $SOURCE @ $ref (expecting $commit)"
echo "target    : $target"

if [ -d "$target/.git" ]; then
    echo "fetching existing clone"
    git -C "$target" fetch --depth 1 origin "$ref" || git -C "$target" fetch origin "$ref" || exit 1
    git -C "$target" checkout -q FETCH_HEAD || exit 1
    git -C "$target" checkout -q -- . || exit 1     # idempotent: drop a previous apply
    git -C "$target" clean -qfd || exit 1            # untracked files a previous apply created; build/ is ignored
else
    [ -e "$target" ] && [ -z "$(ls -A "$target" 2>/dev/null)" ] || [ ! -e "$target" ] || {
        echo "$target exists and is not an empty llama.cpp clone; use --force to remove it" >&2; exit 2; }
    [ "$force" -eq 1 ] && { echo "removing existing $target"; rm -rf "$target"; }
    git clone --depth 1 --branch "$ref" "$SOURCE" "$target" || exit 1
fi

full=$(git -C "$target" rev-parse HEAD)
actual=$(git -C "$target" rev-parse --short HEAD)
if [ -n "$commit" ] && [ "${full#"$commit"}" = "$full" ]; then
    echo "HEAD is $actual ($full), expected a tree starting with $commit - this patch set is generated at" >&2
    echo "exact context, so it would fail or land in the wrong place. Pass --commit (or --ref) for the" >&2
    echo "tree you want." >&2
    exit 1
fi
echo "checked out $actual at $ref"

# Work in a throwaway copy for --check: a patch stack is cumulative, so `patch --dry-run` on a pristine
# tree would test every patch against the wrong file state.
work="$target"
if [ "$check" -eq 1 ]; then
    work=$(mktemp -d)/tree
    cp -a "$target" "$work" || exit 1
    rm -rf "$work/build"
fi

# strip CRLF if the patch files came over from Windows; patch(1) otherwise rejects the context lines
cr=$(mktemp -d)
applied=0
touched=$(mktemp)
for p in "$here"/patches/*.patch; do
    name=$(basename "$p")
    if grep -q $'\r' "$p"; then
        sed -e 's/\r$//' "$p" > "$cr/$name"
        src="$cr/$name"
    else
        src="$p"
    fi
    out=$(cd "$work" && patch -p1 -F0 < "$src" 2>&1) || { echo "FAILED: $name"; echo "$out"; exit 1; }
    grep -qiE 'FAILED|\.rej' <<<"$out" && { echo "REJECTED HUNK in $name"; echo "$out"; exit 1; }
    applied=$((applied + 1))
    grep -oE '^patching file .*' <<<"$out" | sed 's/^patching file //' >> "$touched"
done
echo "applied $applied of $patches patches"
echo "files touched: $(sort -u "$touched" | wc -l)"
if [ "$check" -eq 1 ]; then
    rm -rf "$(dirname "$work")"
    echo "dry run clean - ready to apply into $target"
    exit 0
fi

[ "$build" -eq 1 ] || { echo; echo "ready: cd $target"; exit 0; }

cmake -B "$target/build" -S "$target" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$arch" -DGGML_NATIVE=ON -DGGML_NCCL=OFF -DGGML_CUDA_NCCL=OFF || exit 1
cmake --build "$target/build" -j "$jobs" || exit 1
echo "built: $target/build/bin/llama-bench"
