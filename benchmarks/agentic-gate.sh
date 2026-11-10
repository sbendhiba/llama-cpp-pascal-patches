#!/usr/bin/env bash
# Strict-output gate: the class of failure perplexity and KL cannot see.
#
#   ./agentic-gate.sh --model /path/model.gguf --base-bin /path/stock/build/bin \
#                     --cand-bin /path/patched/build/bin [--port 8099] [--port2 8077]
#
# Each arm serves the model and answers the same 16 prompts at temperature 0 - JSON that has to parse,
# exact tool and git commands, code completions, format constraints - then the two arms' replies are
# compared case by case. The absolute pass rate is not the measurement and should not be quoted as a
# capability number: the validators assume particular key names, so a correct answer spelled differently
# counts as a miss. What matters is that no case passes on one build and fails on the other, and how many
# replies are byte-identical.
#
# Measured bar (2026-11-10, 27B Q6_K_XL, 3x P100 tensor): stock 11/16, this set 11/16, 16/16 replies
# identical, zero CUDA errors on either arm - docs/results.md.
#
# The port handling is not cosmetic and was learned the hard way at 23:14 the same day: a 27 GB
# tensor-split server does not always release its listening socket within a few seconds of SIGKILL, so a
# naive stop-then-start loop lets the next arm's bind fail while /health is still answered by the dying
# previous server. That measures the wrong build and prints perfectly plausible numbers. Hence: a port
# per arm, a hard wait until nothing is listening, and an identity assertion that the PID we launched is
# the one answering and that its own log names the model it was given.
set -uo pipefail

MODEL="" BASE="" CAND="" PORT=8099 PORT2=8077 ARGS="-c 8192 -ub 2048 -fa on -t 28 -sm tensor --parallel 1"
while [ $# -gt 0 ]; do
  case "$1" in
    --model)    MODEL="${2:?}";  shift 2 ;;
    --base-bin) BASE="${2:?}";   shift 2 ;;
    --cand-bin) CAND="${2:?}";   shift 2 ;;
    --port)     PORT="${2:?}";   shift 2 ;;
    --port2)    PORT2="${2:?}";  shift 2 ;;
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

kill_all() { for p in $(ps -eo pid,args | grep -F llama-server | grep -v grep | awk '{print $1}'); do kill -9 "$p" 2>/dev/null; done; }
port_free() { local p="$1" i=0; while [ $i -lt 90 ]; do ss -ltn 2>/dev/null | grep -q ":$p " || return 0; i=$((i+1)); sleep 2; done; return 1; }

arm() { # label bin outfile port
  local label="$1" bin="$2" out="$3" port="$4"
  local lg="/tmp/agentic-$label.log" base=$(basename "$MODEL")
  kill_all
  port_free "$port" || { echo "!! port $port still held - refusing to run $label (it would be answered by the wrong server)" >&2; return 1; }
  echo "== $label $(date -u +%H:%M:%S)"
  "$bin/llama-server" -m "$MODEL" $ARGS --port "$port" --host 127.0.0.1 --no-webui > "$lg" 2>&1 &
  local pid=$! up=0
  for i in $(seq 1 240); do curl -s "http://127.0.0.1:$port/health" 2>/dev/null | grep -q ok && { up=1; break; }; sleep 2; done
  if [ $up -ne 1 ]; then echo "   DOWN: $(tail -3 "$lg" | tr '\n' ' ')" >&2; return 1; fi
  kill -0 "$pid" 2>/dev/null || { echo "   /health answered but our pid $pid is gone - wrong server, aborting $label" >&2; return 1; }
  grep -q -- "$base" "$lg" || { echo "   our log never names $base - aborting $label" >&2; return 1; }
  echo "   up on $port as pid $pid with $base, bind errors $(grep -c "couldn't bind" "$lg")"
  python3 "$PROBE" "$label" "$port" 2> "$out"
  echo "   cuda errors: $(grep -ciE 'cuda error|CUBLAS_STATUS|GGML_ASSERT' "$lg")"
}

D=$(mktemp -d)
arm base "$BASE" "$D/base.cases" "$PORT" || exit 1
arm cand "$CAND" "$D/cand.cases" "$PORT2" || exit 1

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
a, b = load('base'), load('cand')
same = sum(1 for k in a if k in b and a[k][2].strip() == b[k][2].strip())
print('identical replies: %d/%d' % (same, len(a)))
print('base pass / cand fail:', [k for k in a if k in b and a[k][0] == 'PASS' and b[k][0] == 'FAIL'])
print('cand pass / base fail:', [k for k in a if k in b and a[k][0] == 'FAIL' and b[k][0] == 'PASS'])
for k in sorted(a, key=int):
    if k in b and a[k][2].strip() != b[k][2].strip():
        print('-- case %s diverges (%s vs %s)' % (k, a[k][0], b[k][0]))
        print('   base: %s' % a[k][2][:150])
        print('   cand: %s' % b[k][2][:150])
PY
kill_all
rm -rf "$D"
