#!/usr/bin/env bash
# Paired perplexity check, for the arms that change *which reduction runs* rather than how fast it
# runs. `run.sh` measures throughput, and throughput cannot tell "faster" from "different-and-wrong";
# patch 49's whole job is a different reduction, so it needs this one.
#
#   ./ppl.sh --model /path/model.gguf --corpus /path/wiki.test.raw --bin /path/to/build/bin
#
# The corpus is what `scripts/get-wikitext-2.sh` in a llama.cpp tree downloads (wiki.test.raw).
#
# Four arms, same binary, same tokens, same order:
#   tensor-default  -sm tensor with the build's own defaults
#   tensor-optout   ... with GGML_CUDA_AR_PASCAL=0, i.e. the meta-backend fallback on a pre-Volta card
#   tensor-h2d      ... plus GGML_CUDA_AR_H2D_STREAM=1, the second copy-engine stream
#   layer-control   -sm layer, which never issues a cross-device reduce at all
#
# Perplexity is deterministic, so unlike run.sh this needs neither interleaving nor cooling: any
# difference between the rows is the reduction, not the thermal state of the card.
#
# What each row also prints is `init-failed`: the internal AllReduce refuses silently on a card it
# does not accept, and the only sign of that is the warning in the log. A row that says
# init-failed=1 was NOT measuring the internal path, whatever its environment said.
set -uo pipefail

MODEL="" CORPUS="" BINDIR="" CHUNKS=8 CTX=2048 THREADS=28 NGL=99
while [ $# -gt 0 ]; do
  case "$1" in
    --model)   MODEL="${2:?}";   shift 2 ;;
    --corpus)  CORPUS="${2:?}";  shift 2 ;;
    --bin)     BINDIR="${2:?}";  shift 2 ;;
    --chunks)  CHUNKS="${2:?}";  shift 2 ;;
    --ctx)     CTX="${2:?}";     shift 2 ;;
    --threads) THREADS="${2:?}"; shift 2 ;;
    *) echo "ppl.sh: unknown option: $1" >&2; exit 2 ;;
  esac
done
[ -n "$MODEL" ]  || { echo "ppl.sh: --model is required" >&2; exit 2; }
[ -n "$CORPUS" ] || { echo "ppl.sh: --corpus is required" >&2; exit 2; }
BIN="${BINDIR:-.}/llama-perplexity"
[ -x "$BIN" ] || { echo "ppl.sh: not executable: $BIN" >&2; exit 2; }

echo "model : $MODEL"
echo "corpus: $CORPUS ($(wc -c < "$CORPUS") bytes, $CHUNKS chunks of $CTX tokens)"
echo "binary: $BIN"
echo

OUTDIR=$(mktemp -d)
for UB in 512 2048; do
  for ARM in tensor-default tensor-optout tensor-h2d layer-control; do
    o="$OUTDIR/ub${UB}-${ARM}.txt"
    case "$ARM" in
      tensor-default) E=();                          SM="-sm tensor";;
      tensor-optout)  E=(GGML_CUDA_AR_PASCAL=0);     SM="-sm tensor";;
      tensor-h2d)     E=(GGML_CUDA_AR_H2D_STREAM=1); SM="-sm tensor";;
      layer-control)  E=(GGML_CUDA_AR_PASCAL=0);     SM="-sm layer";;
    esac
    env ${E[@]+"${E[@]}"} "$BIN" -m "$MODEL" -f "$CORPUS" -fa on $SM \
        -t "$THREADS" -ngl "$NGL" -c "$CTX" --chunks "$CHUNKS" -ub "$UB" > "$o" 2>&1
    # A layer split builds no comm context at all, so it never logs an init failure and the
    # yes/no below would be a lie; it is the arm with no cross-device reduction to measure.
    if [ "$ARM" = layer-control ]; then
      ar="n/a (no cross-device reduce)"
    elif [ "$(grep -c 'init failed' "$o")" -eq 0 ]; then
      ar=yes
    else
      ar='no (fell back)'
    fi
    printf '%-8s %-16s %s   internal-AllReduce=%s\n' "ub$UB" "$ARM" \
      "$(grep -o 'Final estimate.*' "$o" | tail -1)" "$ar"
  done
  echo
done
echo "raw logs in $OUTDIR"
echo "Reading it: on a pre-Volta card with patch 49, tensor-default is the internal AllReduce and"
echo "tensor-optout is the fallback, so those two rows are the A/B and layer-control is the arm with"
echo "no cross-device reduction at all. On Volta and newer the gate changes nothing, all three tensor"
echo "rows take the same path, and the table is a repeatability check instead."
