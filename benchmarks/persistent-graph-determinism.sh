#!/usr/bin/env bash
# Persistent-kernel determinism under CUDA graph replay.
#
#   ./persistent-graph-determinism.sh /path/to/bin [/path/model.gguf] [cards]
#
# For each arm it starts llama-server fresh, asks the same three greedy questions twice, and prints the
# sha256 of each answer. The claim being tested: a layer half built out of cooperative-grid kernels
# (patches 52-87) must give the same text whether the graph is captured and replayed or every kernel is
# launched directly. Before patch 86 the barrier target was computed on the host and frozen into the
# graph, so replayed barriers stopped synchronising and identical runs diverged.
#
# Pass if the graphs-on column equals the graphs-off column, and both equal the reference build's column.
set -uo pipefail
BIN="${1:?bin dir}"; MODEL="${2:-$MODEL}"; CARDS="${3:-0,1}"
PORT=8123
OUT=$(mktemp -d)
PROMPTS=("The Moon has three notable physical properties:
1." "def merge_sorted(a, b):
" "Explain why water expands when it freezes, using two short paragraphs.")

[ -x "$BIN/llama-server" ] || { echo "not executable: $BIN/llama-server" >&2; exit 2; }

arm() { # tag env...
    local tag="$1"; shift
    local hashes=()
    env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=$CARDS "$@" \
        "$BIN/llama-server" -m "$MODEL" -c 8192 -b 512 -ub 512 -fa on -t 28 -sm layer \
        --parallel 1 --host 127.0.0.1 --port $PORT --no-webui --no-kv-unified \
        > "$OUT/$tag.log" 2>&1 &
    local pid=$!
    for _ in $(seq 1 200); do
        curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break
        kill -0 $pid 2>/dev/null || { echo "$tag: server died, see $OUT/$tag.log" >&2; return 1; }
        sleep 1
    done
    for rep in 1 2; do
        for i in "${!PROMPTS[@]}"; do
            local h
            h=$(curl -s "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
                -d "$(python3 -c '
import json,sys
print(json.dumps({"prompt": sys.argv[1], "n_predict": 128, "temperature": 0, "seed": 42,
                  "cache_prompt": False}))' "${PROMPTS[$i]}")" |
                python3 -c 'import json,sys; sys.stdout.write(json.load(sys.stdin)["content"])' |
                sha256sum | cut -c1-12)
            hashes+=("$h")
        done
    done
    kill $pid 2>/dev/null; wait $pid 2>/dev/null
    grep -qiE 'illegal memory access|CUDA error:' "$OUT/$tag.log" && { echo "$tag: GPU error" >&2; return 1; }
    printf '%-12s %s\n' "$tag" "${hashes[*]}"
    echo "${hashes[*]}" > "$OUT/$tag.set"
}

echo "model: $MODEL"
echo "cards: $CARDS   arms: each 2 fresh processes x 3 prompts, greedy, 128 tokens"
echo
arm graphs-on
arm nographs GGML_CUDA_DISABLE_GRAPHS=1
[ -n "${REF:-}" ] && arm reference
echo
if cmp -s "$OUT/graphs-on.set" "$OUT/nographs.set"; then
    echo "PASS graph replay does not change the greedy output"
else
    echo "FAIL the two arms disagree"
fi
[ -n "${REF:-}" ] && { cmp -s "$OUT/graphs-on.set" "$OUT/reference.set" && echo "PASS equals the reference build" || echo "NOTE differs from the reference build"; }
echo "logs and per-arm hashes in $OUT"
