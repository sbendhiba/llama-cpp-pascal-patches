#!/usr/bin/env bash
# MTP speculative-decoding A/B: stock versus the patch set, four prompts, both arm orders.
#
#   ./mtp-ab.sh --tag v1.0.0-b1b770d --model /path/model.gguf \
#       --base-bin /root/llama.cpp/<ref>-stock/build/bin \
#       --cand-bin /root/llama.cpp/<ref>-patched/build/bin [--reps 2] [--n-gen 256]
#
# The protocol docs/benchmarking.md settled on, in one place instead of re-derived per session:
#
#   - four prompts, not one. A single prompt is a lottery: acceptance depends on how well the MTP
#     head drafts that particular style, and the per-prompt gains in this repo's own table span
#     +45% to +69%. The headline is the mean over the four, and the four go on the page too.
#   - temperature 0 with ignore_eos, so the token sequence is fixed and the arms stay comparable.
#   - both arm orders, each position measured. The arm that runs second in a block is warmer; one
#     order cannot tell that from an effect. The summary prints first-position against
#     second-position for each arm, so a block-order artefact shows up as a gap instead of hiding
#     inside a mean.
#   - cycles per second next to raw t/s. llama.cpp reports draft_n and draft_n_accepted, and the
#     probe turns them into tokens-per-cycle and cycles-per-second (see mtp-probe.py). A throughput
#     number under speculation contains the acceptance rate; the per-cycle one is what the kernels
#     earn, and the two of them together say whether a gain came from the kernels or from luck.
#
# Arms get a port each, a hard wait until `ss -ltn` shows that port free, and an identity check that
# the pid we launched is alive and its own log names the model it was given. That is not decoration:
# a 27 GB tensor-split server does not always release its socket within seconds of SIGKILL, so the
# next arm's bind fails while /health is still answered by the dying previous server - and the arm
# then measures the wrong build while printing entirely plausible numbers. It happened on 2026-11-10
# and cost a whole session before anyone read the logs.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="" BASE="" CAND="" TAG="" REPS=2 NGEN=256
PORT=8099 PORT2=8077
BASE_LABEL="stock" CAND_LABEL="set"
ARGS="-c 8192 -ub 2048 -fa on -t 28 -sm tensor --parallel 1"
SPEC="--spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.2"
BASE_ENV="" CAND_ENV=""
PROMPTS="$HERE/mtp-prompts.txt"

while [ $# -gt 0 ]; do
  case "$1" in
    --model)    MODEL="${2:?}";   shift 2 ;;
    --base-bin) BASE="${2:?}";    shift 2 ;;
    --cand-bin) CAND="${2:?}";    shift 2 ;;
    --tag)      TAG="${2:?}";     shift 2 ;;
    --reps)     REPS="${2:?}";    shift 2 ;;
    --n-gen)    NGEN="${2:?}";    shift 2 ;;
    --args)     ARGS="${2:?}";    shift 2 ;;
    --spec)     SPEC="${2:?}";    shift 2 ;;
    --base-env) BASE_ENV="${2:?}"; shift 2 ;;
    --cand-env) CAND_ENV="${2:?}"; shift 2 ;;
    --base-env)  BASE_ENV="${2:?}";  shift 2 ;;
    --cand-env)  CAND_ENV="${2:?}";  shift 2 ;;
    --base-label) BASE_LABEL="${2:?}"; shift 2 ;;
    --cand-label) CAND_LABEL="${2:?}"; shift 2 ;;
    --prompts)  PROMPTS="${2:?}"; shift 2 ;;
    *) echo "mtp-ab.sh: unknown option: $1" >&2; exit 2 ;;
  esac
done
for v in MODEL BASE CAND TAG; do
  [ -n "${!v}" ] || { echo "mtp-ab.sh: --$(echo "$v" | tr A-Z a-z) is required" >&2; exit 2; }
done
[ -f "$PROMPTS" ] || { echo "mtp-ab.sh: missing prompt set $PROMPTS" >&2; exit 2; }
[ -f "$HERE/mtp-probe.py" ] || { echo "mtp-ab.sh: missing $HERE/mtp-probe.py" >&2; exit 2; }

RAW="$HERE/results/$TAG-mtp.jsonl"
: > "$RAW"
printf 'base=%s\ncand=%s\n' "$BASE_LABEL" "$CAND_LABEL" > "$RAW.meta"
LOG=""
PID=""
ENV_NOW=""

kill_all() { for p in $(ps -eo pid,args | grep -F llama-server | grep -v grep | awk '{print $1}'); do kill -9 "$p" 2>/dev/null; done; }
port_free() { local p="$1" i=0; while [ $i -lt 90 ]; do ss -ltn 2>/dev/null | grep -q ":$p " || return 0; i=$((i+1)); sleep 2; done; return 1; }

start() { # label bin port
  local label="$1" bin="$2" port="$3" base i up
  base=$(basename "$MODEL")
  LOG="/tmp/mtp-$label.log"
  kill_all
  port_free "$port" || { echo "!! port $port still held - refusing to run $label" >&2; return 1; }
  echo "== $label $(date -u +%H:%M:%S)"
  env $ENV_NOW "$bin/llama-server" -m "$MODEL" $ARGS $SPEC --port "$port" --host 127.0.0.1 --no-webui > "$LOG" 2>&1 &
  PID=$!
  up=0
  for i in $(seq 1 240); do
    curl -s "http://127.0.0.1:$port/health" 2>/dev/null | grep -q ok && { up=1; break; }
    sleep 2
  done
  [ "$up" -eq 1 ] || { echo "   DOWN: $(tail -3 "$LOG" | tr '\n' ' ')" >&2; return 1; }
  kill -0 "$PID" 2>/dev/null || { echo "   /health answered but pid $PID is gone - wrong server" >&2; return 1; }
  grep -q -- "$base" "$LOG" || { echo "   our log never names $base - aborting $label" >&2; return 1; }
  echo "   up on $port as pid $PID with $base, bind errors $(grep -c "couldn't bind" "$LOG")"
}

run_arm() { # arm order pos bin port
  local arm="$1" order="$2" pos="$3" bin="$4" port="$5" rep lab
  lab="$arm-$order-$pos"
  if [ "$arm" = base ]; then ENV_NOW="$BASE_ENV"; else ENV_NOW="$CAND_ENV"; fi
  start "$lab" "$bin" "$port" || return 1
  for rep in $(seq 1 "$REPS"); do
    python3 "$HERE/mtp-probe.py" "$port" "$PROMPTS" "$NGEN" "$lab" "$rep" >> "$RAW"
  done
  echo "   cuda errors: $(grep -ciE 'cuda error|CUBLAS_STATUS|GGML_ASSERT' "$LOG")"
}

for order in ab ba; do
  echo "--- order $order ---"
  if [ "$order" = ab ]; then
    run_arm base ab 1 "$BASE" "$PORT"  || { kill_all; exit 1; }
    run_arm cand ab 2 "$CAND" "$PORT2" || { kill_all; exit 1; }
  else
    run_arm cand ba 1 "$CAND" "$PORT2" || { kill_all; exit 1; }
    run_arm base ba 2 "$BASE" "$PORT"  || { kill_all; exit 1; }
  fi
done
kill_all

python3 - "$RAW" <<'PY'
import json
import os
import sys

labels = {"base": "stock", "cand": "set"}
if os.path.exists(sys.argv[1] + ".meta"):
    for line in open(sys.argv[1] + ".meta"):
        if "=" in line:
            k, v = line.strip().split("=", 1)
            labels[k] = v

rows = []
for line in open(sys.argv[1], errors="ignore"):
    line = line.strip()
    if not line or not line.startswith("{"):
        continue
    try:
        r = json.loads(line)
    except ValueError:
        continue
    if r.get("error"):
        print("!! %s %s: %s" % (r["label"], r.get("prompt"), r["error"]))
        continue
    if "label" not in r:
        continue
    rows.append(r)


def sel(arm=None, prompt=None, pos=None):
    out = []
    for r in rows:
        lab = r["label"].split("-")
        if arm and lab[0] != arm:
            continue
        if pos and lab[2] != pos:
            continue
        if prompt and r.get("prompt") != prompt:
            continue
        out.append(r)
    return out


def mean(vals):
    vals = [v for v in vals if v]
    return sum(vals) / len(vals) if vals else None


prompts = []
for r in rows:
    if r.get("prompt") not in prompts:
        prompts.append(r["prompt"])
prompts.sort()

print("%-11s %9s %9s %8s | %9s %9s %8s" % ("prompt", labels["base"], labels["cand"], "gain",
                                            labels["base"] + "/cyc", labels["cand"] + "/cyc", "gain"))
for p in prompts:
    b, c = sel("base", p), sel("cand", p)
    if not b or not c:
        print("%-11s  (missing arm: %d stock rows, %d set rows)" % (p, len(b), len(c)))
        continue
    bpps, cpps = mean([r["pps"] for r in b]), mean([r["pps"] for r in c])
    bc, cc = mean([r.get("per_cycle") for r in b]), mean([r.get("per_cycle") for r in c])
    line = "%-11s %9.2f %9.2f %+7.1f%%" % (p, bpps, cpps, 100 * (cpps / bpps - 1))
    if bc and cc:
        line += " | %9.2f %9.2f %+7.1f%%" % (bc, cc, 100 * (cc / bc - 1))
    print(line)

b_all, c_all = sel("base"), sel("cand")
bp, cp = mean([r["pps"] for r in b_all]), mean([r["pps"] for r in c_all])
bc, cc = mean([r.get("per_cycle") for r in b_all]), mean([r.get("per_cycle") for r in c_all])
bl, cl = mean([r.get("mean_len") for r in b_all]), mean([r.get("mean_len") for r in c_all])
line = "%-11s %9.2f %9.2f %+7.1f%%" % ("MEAN", bp, cp, 100 * (cp / bp - 1))
if bc and cc:
    line += " | %9.2f %9.2f %+7.1f%%" % (bc, cc, 100 * (cc / bc - 1))
print(line)
if bl and cl:
    print("mean tokens per cycle: %s %.2f, %s %.2f  (a gap here means the arms drafted differently, "
          "not that one is faster)" % (labels["base"], bl, labels["cand"], cl))

print("--- position within the block (clock sag check) ---")
for arm in ("base", "cand"):
    parts = []
    for pos in ("1", "2"):
        m = mean([r["pps"] for r in sel(arm, pos=pos)])
        if m:
            parts.append("pos%s %.2f" % (pos, m))
    if parts:
        print("  %-5s %s" % (arm, ", ".join(parts)))

print("raw rows: %s (%d)" % (sys.argv[1], len(rows)))
PY
