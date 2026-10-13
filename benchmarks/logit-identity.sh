#!/usr/bin/env bash
# Logit-level output identity for patch 49's two increments. The repo has paired perplexity
# (benchmarks/ppl.sh) but no logit-level check, and perplexity to 4 decimals is a weak instrument
# for a synchronisation change. This one uses llama-perplexity's --save-all-logits and --kl-divergence:
# the reference arm writes its logits, every other arm scores KLD against that same file.
#
#   ./logit-identity.sh /path/model.gguf /path/wiki.test.raw /path/to/build/bin [n_ctx] [ub]
#
# KLD = 0.000000 with max_kl 0 means the two runs produced bit-identical logits.
set -uo pipefail

MODEL="${1:?model}"; CORPUS="${2:?corpus}"; BIN="${3:?bin dir}"
CTX="${4:-512}"; UB="${5:-512}"
OUT="$(mktemp -d)"
P="$BIN/llama-perplexity"
[ -x "$P" ] || { echo "not executable: $P" >&2; exit 2; }

# exchanges must be over the copy-engine threshold (1 MB) for the H2D stream to matter:
# n_hidden/2 x ub x 2 bytes per card, so ub 512 on a 5120-hidden model is ~2.6 MB.
run() { # tag env...
  local tag="$1"; shift
  env "$@" "$P" -m "$MODEL" -f "$CORPUS" -fa on -sm tensor -t 28 -ngl 99 \
      -c "$CTX" --chunks 1 -ub "$UB" --save-all-logits "$OUT/$tag.gguf" > "$OUT/$tag.txt" 2>&1
  printf '%-14s %s   logits=%s\n' "$tag" \
    "$(grep -oE 'Final estimate.*' "$OUT/$tag.txt" | tail -1)" \
    "$(sha256sum "$OUT/$tag.gguf" 2>/dev/null | cut -c1-16)"
}

echo "model : $MODEL"
echo "corpus: $CORPUS   ctx=$CTX ub=$UB   (1 chunk, all logits exported)"
echo

run reference
for arm in optout h2d ngpu; do
  case "$arm" in
    optout) E=(GGML_CUDA_AR_PASCAL=0);;
    h2d)    E=(GGML_CUDA_AR_H2D_STREAM=1);;
    ngpu)   E=(GGML_CUDA_AR_COPY_CHUNK_BYTES=262144);;
  esac
  env "${E[@]}" "$P" -m "$MODEL" -f "$CORPUS" -fa on -sm tensor -t 28 -ngl 99 \
      -c "$CTX" --chunks 1 -ub "$UB" --kl-divergence --kl-divergence-base "$OUT/reference.gguf" \
      > "$OUT/$arm.txt" 2>&1
  printf '%-14s %s\n' "$arm" "$(grep -iE 'KL divergence|kl_div' "$OUT/$arm.txt" | tail -2 | tr '\n' ' ')"
  grep -oE 'Final estimate.*' "$OUT/$arm.txt" | tail -1 | sed 's/^/               /'
done

echo
echo "raw logs and logits in $OUT"
