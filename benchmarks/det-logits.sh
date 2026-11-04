#!/usr/bin/env bash
# Greedy reproducibility, decided on the logits.
#
#   ./det-logits.sh /path/to/build/bin [model.gguf] [corpus.txt] [cards]
#
# Why the logits and not the text: a greedy argmax hides every change below the margin between the top
# two tokens, and llama-cli's stdout carries its own timing block - a probe that hashes stdout reports
# three different digests for three runs of the *reference* arm and tells you nothing. llama-perplexity
# never samples, so what it writes with --save-all-logits is the function's output, at every position.
#
# Two batch shapes, because different patches own them: one 512-token chunk is the prefill path, and a
# run of 4-token chunks is the 2..6-token verify path the MTP head uses. A patch that reorders work
# between nodes shows up in the second one and not the first, which is exactly how patch 24 was found.
#
# Pass: the default repeats itself (one digest across three runs) and graphs-off agrees with it.
# Different is not automatically wrong - the block being on changes the arithmetic deliberately - but
# different *between two identical runs* is a race, and a race is not shippable as a default.
set -u
export NO_COLOR=1
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BIN="${1:?bin dir (the one under test)}"
MODEL="${2:-$MODEL}"
CORPUS="${3:-$HERE/det-corpus.txt}"
CARDS="${4:-0,1}"
CTX="${CTX:-256}"
OUT=$(mktemp -d)

[ -x "$BIN/llama-perplexity" ] || { echo "not executable: $BIN/llama-perplexity" >&2; exit 2; }
[ -r "$CORPUS" ] || { echo "no corpus at $CORPUS" >&2; exit 2; }
echo "bin $BIN"
echo "model $(basename "$MODEL")   corpus $CORPUS ($(wc -c < "$CORPUS") bytes)   cards $CARDS   ctx $CTX"

run() { # tag shape-args env...
    local tag="$1" shape="$2"; shift 2
    env "$@" CUDA_VISIBLE_DEVICES="$CARDS" "$BIN/llama-perplexity" -m "$MODEL" -f "$CORPUS" \
        -fa on -t 28 -ngl 99 -sm layer -c "$CTX" $shape --save-all-logits "$OUT/$tag.gguf" \
        > "$OUT/$tag.log" 2>&1
    printf '  %-14s rc=%-3s %s\n' "$tag" "$?" "$(grep -oE 'Final estimate.*' "$OUT/$tag.log" | tail -1)"
}

echo
echo "### verify shape: -ub 4, 64 chunks of 4 tokens"
run default-1 "-ub 4 --chunks 64"
run default-2 "-ub 4 --chunks 64"
run default-3 "-ub 4 --chunks 64"
run nographs  "-ub 4 --chunks 64" GGML_CUDA_DISABLE_GRAPHS=1

echo "### prefill shape: -ub 256, one chunk"
run default-pf-1 "-ub 256 --chunks 1"
run default-pf-2 "-ub 256 --chunks 1"
run nographs-pf  "-ub 256 --chunks 1" GGML_CUDA_DISABLE_GRAPHS=1

echo
python3 "$HERE/det-summarize.py" "$OUT"
