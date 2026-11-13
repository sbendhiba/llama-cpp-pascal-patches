#!/usr/bin/env bash
# Greedy reproducibility, decided on the logits.
#
#   ./det-logits.sh /path/to/build/bin [model.gguf] [corpus.txt] [device-list]
#
# The fourth argument is a CUDA_VISIBLE_DEVICES **list** - "0,1,2" - not a card count. Passing "3"
# selects the fourth device of a three-GPU box, llama.cpp answers "no CUDA-capable device is detected",
# and the whole gate runs on the CPU: deterministic, quiet, and proving nothing about the CUDA path it
# exists to test. The arm assertion below is what that cost, written so it cannot happen quietly again.
#
# Why the logits and not the text: a greedy argmax hides every change below the margin between the top
# two tokens, and llama-cli's stdout carries its own timing block - a probe that hashes stdout reports
# three different digests for three runs of the *reference* arm and tells you nothing. llama-perplexity
# never samples, so what it writes with --save-all-logits is the function's output, at every position.
#
# Two batch shapes, because different patches own them: one 512-token chunk is the prefill path, and a
# run of 4-token chunks is the 2..6-token verify path the MTP head uses. A patch that reorders work
# between nodes shows up in the second one and not the first, which is exactly how patch 24 was found.
# Both shapes run `-sm layer`, which is also the only configuration in which patch 22's decode slots are
# live, so this probe is the slots' determinism test as well.
#
# Cost, since --chunks is passes over the corpus and not tokens: at -ub 4 a 256-token pass costs tens of
# seconds on three P100s and the verify shape runs 64 of them, four times over. Use the shipped
# det-corpus.txt (VCHUNKS/PFCHUNKS shorten it further if you need a quick look); pointing this at a whole
# WikiText file is an hours-long run, not a smoke test.
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
VCHUNKS="${VCHUNKS:-64}"
PFCHUNKS="${PFCHUNKS:-1}"
OUT=$(mktemp -d)

[ -x "$BIN/llama-perplexity" ] || { echo "not executable: $BIN/llama-perplexity" >&2; exit 2; }
[ -r "$CORPUS" ] || { echo "no corpus at $CORPUS" >&2; exit 2; }
case "$CARDS" in
    *,*|0|1|2|3) ;;
    *) echo "det-logits.sh: cards should be a comma-separated CUDA device list (0,1,2), got '$CARDS'" >&2; exit 2 ;;
esac
echo "bin $BIN"
echo "model $(basename "$MODEL")   corpus $CORPUS ($(wc -c < "$CORPUS") bytes)   devices $CARDS   ctx $CTX"

run() { # tag shape-args env...
    local tag="$1" shape="$2"; shift 2
    env "$@" CUDA_VISIBLE_DEVICES="$CARDS" "$BIN/llama-perplexity" -m "$MODEL" -f "$CORPUS" \
        -fa on -t 28 -ngl 99 -sm layer -c "$CTX" $shape --save-all-logits "$OUT/$tag.gguf" \
        > "$OUT/$tag.log" 2>&1
    local rc=$?
    # An arm that fell off the GPU is not an arm. Check the device line the binary itself prints.
    if ! grep -q "CUDA : ARCHS" "$OUT/$tag.log" || grep -q "no CUDA-capable device" "$OUT/$tag.log"; then
        echo "  $tag DID NOT RUN ON THE GPU: $(grep -oE 'ggml_cuda_init: [^|]*' "$OUT/$tag.log" | head -1)" >&2
        echo "  CUDA_VISIBLE_DEVICES='$CARDS' - fix the device list, this gate means nothing on the CPU" >&2
        exit 2
    fi
    printf '  %-14s rc=%-3s %s\n' "$tag" "$rc" "$(grep -oE 'Final estimate.*' "$OUT/$tag.log" | tail -1)"
}

echo
echo "### verify shape: -ub 4, $VCHUNKS chunks of 4 tokens"
run default-1 "-ub 4 --chunks $VCHUNKS"
run default-2 "-ub 4 --chunks $VCHUNKS"
run default-3 "-ub 4 --chunks $VCHUNKS"
run nographs  "-ub 4 --chunks $VCHUNKS" GGML_CUDA_DISABLE_GRAPHS=1

echo "### prefill shape: -ub 256, $PFCHUNKS chunk"
run default-pf-1 "-ub 256 --chunks $PFCHUNKS"
run default-pf-2 "-ub 256 --chunks $PFCHUNKS"
run nographs-pf  "-ub 256 --chunks $PFCHUNKS" GGML_CUDA_DISABLE_GRAPHS=1

echo
python3 "$HERE/det-summarize.py" "$OUT"
