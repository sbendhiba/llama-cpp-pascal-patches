#!/usr/bin/env bash
# Strict-output gate: the class of failure perplexity and KL cannot see.
#
#   ./agentic-gate.sh --model /path/model.gguf --base-bin /path/stock/build/bin \
#                     --cand-bin /path/patched/build/bin [--port 8099]
#
# Each arm serves the model with MTP off and answers the same 16 prompts at temperature 0 - JSON that has
# to parse, exact tool and git commands, code completions, format constraints - then the two arms' replies
# are compared case by case. The absolute pass rate is not the measurement and should not be quoted as a
# capability number: the validators assume particular key names, so a correct answer spelled differently
# counts as a miss. What matters is that no case passes on one build and fails on the other, and how many
# replies are byte-identical.
#
# Measured bar (2026-11-10, 27B Q6_K_XL, 3x P100 tensor): stock 11/16, this set 11/16, 16/16 replies
# identical, zero CUDA errors on either arm - docs/results.md.
set -uo pipefail

MODEL="" BASE="" CAND="" PORT=8099 ARGS="-c 8192 -ub 2048 -fa on -t 28 -sm tensor --parallel 1"
while [ $# -gt 0 ]; do
  case "$1" in
    --model)    MODEL="${2:?}";  shift 2 ;;
    --base-bin) BASE="${2:?}";   shift 2 ;;
    --cand-bin) CAND="${2:?}";   shift 2 ;;
    --port)     PORT="${2:?}";   shift 2 ;;
    --args)     ARGS="${2:?}";   shift 2 ;;
    *) echo "agentic-gate.sh: unknown option: $1" >&2; exit 2 ;;
  esac
done
for v in MODEL BASE CAND; do
  [ -n "${!v}" ] || { echo "agentic-gate.sh: --$(echo "$v" | tr A-Z a-z | sed 's/-bin//') is required" >&2; exit 2; }
done
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBE="$HERE/agentic-probe.py"
[ -f "$PROBE" ] || { echo "agentic-gate.sh: missing $PROBE" >&2; exit 2; }

stop() { for p in $(ps -eo pid,args | grep -F llama-server | grep -v grep | awk '{print $1}'); do kill -9 "$p" 2>/dev/null; done; sleep 3; }
arm() { # label bin outfile
  local label="$1" bin="$2" out="$3"
  stop
  echo "== $label $(date -u +%H:%M:%S)"
  "$bin/llama-server" -m "$MODEL" $ARGS --port "$PORT" --no-webui > "/tmp/ag-$label.log" 2>&1 &
  local up=0
  for i in $(seq 1 200); do
    curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok && { up=1; break; }
    sleep 2
  done
  if [ $up -ne 1 ]; then echo "   server did not come up:"; tail -5 "/tmp/ag-$label.log"; return 1; fi
  python3 "$PROBE" "$label" 2> "$out"
  echo "   cuda errors: $(grep -ciE 'cuda error|CUBLAS_STATUS|GGML_ASSERT' "/tmp/ag-$label.log")"
  stop
}

D=$(mktemp -d)
arm stock "$BASE" "$D/stock.cases"
arm cand  "$CAND" "$D/cand.cases"

python3 - "$D" <<'PY'
import os, sys
d = sys.argv[1]
def load(lab):
    out = {}
    for line in open(os.path.join(d, '%s.cases' % lab), errors='ignore'):
        if line.startswith('CASE '):
            p = line.split(None, 4)
            out[p[1]] = (p[2], p[3], p[4] if len(p) > 4 else '')
    return out
a, b = load('stock'), load('cand')
same = sum(1 for k in a if k in b and a[k][2].strip() == b[k][2].strip())
print('identical replies: %d/%d' % (same, len(a)))
print('stock pass / cand fail:', [k for k in a if k in b and a[k][0] == 'PASS' and b[k][0] == 'FAIL'])
print('cand pass / stock fail:', [k for k in a if k in b and a[k][0] == 'FAIL' and b[k][0] == 'PASS'])
for k in sorted(a, key=int):
    if k in b and a[k][2].strip() != b[k][2].strip():
        print('-- case %s diverges (%s vs %s)' % (k, a[k][0], b[k][0]))
        print('   stock: %s' % a[k][2][:150])
        print('   cand : %s' % b[k][2][:150])
PY
rm -rf "$D"
