#!/usr/bin/env bash
# Benchmark protocol runner for the Pascal patch set.
#
# Runs N rounds with the arms interleaved and the arm order reversed on alternate rounds, which is
# the protocol docs/benchmarking.md settled on after two measurements were wrong because the arms
# were not rotated. Each arm is a different llama.cpp build (or one build with a patch's environment
# variable toggled), so the only difference between rows is the arm.
#
#   ./run.sh --tag v0.6.0 --gpu p100x2 --reps 5 \
#       --arm stock=/opt/llama-v060/build/bin \
#       --arm patched=/root/llama.cpp/build/bin \
#       -- -m /root/models/Qwen3.8-27B-Q4_0.gguf -fa on -sm tensor -t 28 -p 512,2048 -n 128
#
# Everything after -- is passed to llama-bench verbatim; the script adds "-r 1 -o jsonl" per call so
# each round is one sample per test and the interleaving is real.
#
# Results append to results/<tag>-<gpu>.csv (one row per arm, round and test, with the GPU temperature
# it was taken at). A markdown summary goes to stdout and to results/<tag>-<gpu>-summary.md.
#
# Two things this script learned the hard way on 2026-10-11, when a full 10-round run finished cold
# and wrote an empty file:
#
# - A llama-bench jsonl record carries **no test name**. `test::get_fields()` ends at `stddev_ts`;
#   the `pp2048` / `tg128` label only exists in the md and csv printers, where it is built from
#   `n_prompt`, `n_gen` and `n_depth`. `test_name` below rebuilds it the same way.
# - Silence is not a safe default. If a call returns no records, or a record has no `avg_ts`, the
#   script now stops on the first round instead of burning the rest of them to produce nothing.
#
# What this script cannot do: cool your card. Idle until the GPU is back under the temperature you
# chose before starting - a systematically warmer arm silently loses, and on a P100 the boost sag
# across a long run (1328 -> 1101 MHz) is larger than most effects here.
set -euo pipefail

TAG="" GPU="" REPS=5
OUTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results"
ARMS=() BENCH=()

die() { echo "run.sh: $*" >&2; exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --tag)         TAG="${2:?}"; shift 2 ;;
    --gpu)         GPU="${2:?}"; shift 2 ;;
    --reps)        REPS="${2:?}"; shift 2 ;;
    --arm)         ARMS+=("${2:?}"); shift 2 ;;
    --results-dir) OUTDIR="${2:?}"; shift 2 ;;
    -h|--help)     sed -n '2,31p' "${BASH_SOURCE[0]}"; exit 0 ;;
    --)            shift; BENCH=("$@"); break ;;
    *)             die "unknown option: $1" ;;
  esac
done

[ -n "$TAG" ] || die "--tag is required (the llama.cpp version under test)"
[ -n "$GPU" ] || die "--gpu is required, e.g. p100x2 (it names the output file)"
[ ${#ARMS[@]} -ge 1 ] || die "give at least one --arm name=/path/to/build/bin"
[ ${#BENCH[@]} -ge 1 ] || die "give llama-bench arguments after --"
mkdir -p "$OUTDIR"
CSV="$OUTDIR/$TAG-$GPU.csv"
SUMMARY="$OUTDIR/$TAG-$GPU-summary.md"
HDR='round,arm,when,tag,gpu,build,gpu_temp_c,test,avg_ts,stddev_ts,n_prompt,n_gen,env,bench_args'
START="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
BASE="${ARMS[0]%%=*}"

# one jsonl field out of a llama-bench record: quoted string, number, or the nan/inf a single
# sample produces (llama-bench snprintfs stddev_ts, so it is bare and unquoted in the record)
field() {
  printf '%s' "$1" \
    | grep -o "\"$2\": \"[^\"]*\"\|\"$2\": [-+0-9.eEinfa]*" | head -1 \
    | sed "s/^\"$2\": //; s/\"//g" || true
}

# llama-bench's jsonl record has no test name; rebuild the one its md/csv printers would show.
test_name() {
  local np ng nd name
  np="$(field "$1" n_prompt)"
  ng="$(field "$1" n_gen)"
  nd="$(field "$1" n_depth)"
  name=""
  if [ "${np:-0}" != "0" ] && [ "${ng:-0}" = "0" ]; then
    name="pp${np}"
  elif [ "${ng:-0}" != "0" ] && [ "${np:-0}" = "0" ]; then
    name="tg${ng}"
  elif [ "${np:-0}" != "0" ]; then
    name="pp${np}+tg${ng}"
  else
    name="other"
  fi
  if [ "${nd:-0}" != "0" ]; then
    name="${name}@d${nd}"
  fi
  printf '%s' "$name"
}

temp_c() {
  command -v nvidia-smi >/dev/null 2>&1 || { printf 'na'; return 0; }
  nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null \
    | awk '{s += $1; n++} END {if (n) printf "%.0f", s/n; else printf "na"}'
}

env_capture() {
  local v
  v=$(env | grep -E '^(GGML_|LLAMA_|CUDA_|GP100_)' | cut -d= -f1 | sort | paste -sd ';' -) || true
  printf '%s' "${v:-none}"
}

if [ ! -f "$CSV" ]; then
  echo "$HDR" > "$CSV"
elif [ "$(head -1 "$CSV")" != "$HDR" ]; then
  die "$CSV has a different header than this script writes; move it aside rather than appending"
elif [ "$(wc -l < "$CSV")" -gt 1 ]; then
  echo "note: $CSV already holds $(( $(wc -l < "$CSV") - 1 )) row(s) from earlier runs; they stay in" \
       "the file but the summary below only covers this run." >&2
fi

printf '== %s on %s: %s round(s), %s arm(s)\n' "$TAG" "$GPU" "$REPS" "${#ARMS[@]}"
printf '== env: %s\n'  "$(env_capture)"
printf '== args: %s\n\n' "${BENCH[*]}"

for ((r = 1; r <= REPS; r++)); do
  # alternate arm order so neither arm systematically runs on the warmer card
  order=()
  for ((i = 0; i < ${#ARMS[@]}; i++)); do order+=("$i"); done
  if [ $((r % 2)) -eq 0 ]; then
    rev=(); for ((i = ${#order[@]} - 1; i >= 0; i--)); do rev+=("${order[i]}"); done; order=("${rev[@]}")
  fi

  for i in "${order[@]}"; do
    spec="${ARMS[$i]}"; name="${spec%%=*}"; bindir="${spec#*=}"
    bin="$bindir/llama-bench"
    [ -x "$bin" ] || die "not executable: $bin"

    t="$(temp_c)"
    printf '   round %s/%s  arm %-14s  %s C\n' "$r" "$REPS" "$name" "$t"

    out="$("$bin" -r 1 -o jsonl "${BENCH[@]}" 2>/dev/null)" || die "llama-bench failed for arm '$name'"

    nrec=0
    while IFS= read -r line; do
      case "$line" in *'"samples_ts"'*) ;; *) continue ;; esac
      avg="$(field "$line" avg_ts)"
      [ -n "$avg" ] || die "a record for arm '$name' carries no avg_ts; refusing to write a blank row"
      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"%s","%s"\n' \
        "$r" "$name" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$TAG" "$GPU" \
        "$(field "$line" build_commit)" "$t" \
        "$(test_name "$line")" "$avg" "$(field "$line" stddev_ts)" \
        "$(field "$line" n_prompt)" "$(field "$line" n_gen)" \
        "$(env_capture)" "${BENCH[*]}" >> "$CSV"
      nrec=$((nrec + 1))
    done <<< "$out"

    [ "$nrec" -gt 0 ] || die "llama-bench returned no records for arm '$name' (round $r); its stdout \
began: $(printf '%s' "$out" | head -c 300)"
  done
done

echo
# Mean and spread across the rounds of THIS run. llama-bench's own stddev_ts is meaningless here
# because every call is -r 1, so it is stored raw in the csv and recomputed from the round values.
awk -F, -v start="$START" -v base="$BASE" '
  NR == 1 { next }
  $3 < start { next }
  {
    k = $2 SUBSEP $8
    v = $9 + 0
    sum[k] += v; cnt[k]++; sq[k] += v * v
    arm[$2] = 1; test[$8] = 1; seen[$2] = 1
    if ($7 ~ /^[0-9]+$/) {
      if (!($2 in tmin) || $7 + 0 < tmin[$2]) tmin[$2] = $7 + 0
      if (!($2 in tmax) || $7 + 0 > tmax[$2]) tmax[$2] = $7 + 0
    }
  }
  END {
    printf "| arm | test | mean t/s | sd across rounds | rounds | temps C |\n"
    printf "|---|---|---:|---:|---:|---|\n"
    for (a in arm) for (t in test) {
      k = a SUBSEP t
      if (!(k in sum)) continue
      m = sum[k] / cnt[k]
      sd = "na"
      if (cnt[k] > 1) {
        var = (sq[k] - cnt[k] * m * m) / (cnt[k] - 1)
        if (var < 0) var = 0
        sd = sprintf("%.2f", sqrt(var))
      }
      tr = (a in tmin) ? tmin[a] "-" tmax[a] " C" : "na"
      printf "| %s | %s | %.2f | %s | %d | %s |\n", a, t, m, sd, cnt[k], tr
    }
    if (!(base in seen)) { print ""; print "no rows for the base arm " base; exit }
    printf "\ndelta of each other arm against %s:\n", base
    for (a in arm) for (t in test) {
      if (a == base || !((a SUBSEP t) in sum) || !((base SUBSEP t) in sum)) continue
      b = sum[base SUBSEP t] / cnt[base SUBSEP t]
      if (b > 0) printf "   %s on %s: %+.2f%%\n", a, t, \
        100 * ((sum[a SUBSEP t] / cnt[a SUBSEP t]) / b - 1)
    }
  }' "$CSV" | tee "$SUMMARY"

echo
echo "wrote $CSV and $SUMMARY"
echo "Then: put the numbers in the README table with the date, the llama.cpp tag and the arm"
echo "definitions. A throughput figure without its build commit, its temperatures and its arm"
echo "ordering is not a measurement - see docs/benchmarking.md for why."
