#!/usr/bin/env bash
# The quality gate for the set as a package: perplexity, and logit KL against the build it replaces.
#
#   ./kld-gate.sh --model /path/model.gguf --corpus /path/wiki.test.raw \
#                 --base-bin /path/stock/build/bin --cand-bin /path/patched/build/bin
#
# Why package-level. What gets installed is this directory of patches, not one patch, so the question is
# "does the set as a whole still compute what the build it replaces computes?" Per-patch attribution is
# debugging, not the gate: at a prefill batch shape the decode kernels are not reached at all
# (ggml-cuda.cu routes mul_mat_vec_q only while src1->ne[1] <= MMVQ_MAX_BATCH_SIZE, which is 8), so a
# KL measured there says nothing about them, and a KL that does not move when a decode patch is switched
# off is a statement about routing, not about quality.
#
# Three shapes, because which kernels answer depends on the batch width (see the protocol in
# CONTRIBUTING.md - perplexity is only meaningful for these patches when the decode path is actually
# taken):
#
#   prefill  -b 2048 -ub 512   MMQ and cuBLAS. What llama.cpp normally does when you feed it a document.
#   verify   -b 8    -ub 5      the MTP speculative verify width, so the decode patches do run.
#   decode   -b 8    -ub 1      single-token decode, the other half of what those patches serve.
#
# For each shape: the accepted build runs and writes its logits, the candidate runs scored against them,
# and one arm scores the accepted build against its own logits. That last arm is the harness floor and it
# has to read 0.000000; if it does not, --save-all-logits or the corpus moved and nothing in the report
# means anything.
#
# How to read the numbers:
#   * PPL is the ground-truth metric. If the candidate is worse by more than about 0.5% on a quant you
#     ship, stop and find out why before merging.
#   * KL needs a scale, and the scale is a quantization step on this same harness: UD-Q4_K_M against
#     Q8_0 of the same model is mean KLD 0.0306 +- 0.0051, and UD-Q6_K_XL against Q8_0 is
#     0.0103 +- 0.0026 (docs/results.md, 2026-11-10). The 79-patch set measures 0.0066 and 0.0102 against
#     stock at the prefill shape - inside the difference between two quantizations of the same weights,
#     which is what people already accept without argument. Re-measure the reference if you want to quote
#     it at a new ref.
#   * Maximum KLD is expected to be large even when nothing is wrong: a small early difference is
#     amplified chaotically through a 2048-token chunk, so a few positions change their mind completely.
#     Judge the mean and the 95th percentile; use the maximum only to compare two candidates.
set -uo pipefail

MODEL="" CORPUS="" BASE="" CAND="" CTX=2048 THREADS=28 NGL=99 WORK="" CHUNKS_PF=8 CHUNKS_DEC=2
while [ $# -gt 0 ]; do
  case "$1" in
    --model)    MODEL="${2:?}";  shift 2 ;;
    --corpus)   CORPUS="${2:?}"; shift 2 ;;
    --base-bin) BASE="${2:?}";   shift 2 ;;
    --cand-bin) CAND="${2:?}";   shift 2 ;;
    --chunks-prefill) CHUNKS_PF="${2:?}"; shift 2 ;;
    --chunks-decode)  CHUNKS_DEC="${2:?}"; shift 2 ;;
    --ctx)      CTX="${2:?}";    shift 2 ;;
    --threads)  THREADS="${2:?}"; shift 2 ;;
    --work)     WORK="${2:?}";   shift 2 ;;
    *) echo "kld-gate.sh: unknown option: $1" >&2; exit 2 ;;
  esac
done
for v in MODEL CORPUS BASE CAND; do
  [ -n "${!v}" ] || { echo "kld-gate.sh: --$(echo "$v" | tr A-Z a-z | sed 's/-bin//') is required" >&2; exit 2; }
done
for b in "$BASE/llama-perplexity" "$CAND/llama-perplexity"; do
  [ -x "$b" ] || { echo "kld-gate.sh: not executable: $b" >&2; exit 2; }
done
WORK="${WORK:-/root/pplq}"
mkdir -p "$WORK" || exit 2
echo "logits go in $WORK - roughly 1 GB per 4096 tokens, and /tmp is tmpfs on the bench box, so pick a disk path"

# tag shape-batch shape-ubatch chunks bin extra...
run() {
  local tag="$1" bs="$2" ub="$3" chunks="$4" bin="$5"; shift 5
  "$bin/llama-perplexity" -m "$MODEL" -f "$CORPUS" -c "$CTX" --chunks "$chunks" -ngl "$NGL" \
      -sm tensor -fa on -t "$THREADS" -b "$bs" -ub "$ub" "$@" > "$WORK/$tag.log" 2>&1
  local rc=$?
  # the minus in -0.000000 is real and it is the floor: llama-perplexity prints the mean of per-position
  # KL divergences, and a sum that lands a hair below zero prints that way. A matcher without [-+] reads
  # the one arm whose job is to prove the harness is honest as a blank, which is how this gate reported
  # "floor-decode: (nothing)" on v1.0.0 while the floor was exactly zero.
  printf '%-16s rc=%-3s %-38s | %-32s | %s\n' "$tag" "$rc" \
    "$(grep -oE 'Final estimate: PPL = [-+0-9][0-9.]* \+/- [-+0-9][0-9.]*' "$WORK/$tag.log" | tail -1)" \
    "$(grep -oE 'Mean    KLD: *[-+0-9][0-9.]* ± *[0-9.]+' "$WORK/$tag.log" | tail -1)" \
    "$(grep -oE '95.0%   KLD: *[-+0-9][0-9.]*' "$WORK/$tag.log" | tail -1)"
  [ $rc -ne 0 ] && tail -3 "$WORK/$tag.log"
}

echo
echo "model : $MODEL"
echo "corpus: $CORPUS   ctx $CTX   -sm tensor   ngl $NGL   threads $THREADS"
echo "base  : $BASE"
echo "cand  : $CAND"
echo
echo "--- prefill (-b 2048 -ub 512, $CHUNKS_PF chunks) ---"
run base-prefill     2048 512 "$CHUNKS_PF" "$BASE" --save-all-logits "$WORK/kld-base-prefill.gguf"
run cand-prefill     2048 512 "$CHUNKS_PF" "$CAND" --kl-divergence --kl-divergence-base "$WORK/kld-base-prefill.gguf"
# The candidate scored against base logits prints the KLD table and no PPL of its own, so the column the
# README's quality table quotes - "PPL this set" - came from a separate plain run each time it was written.
# Making it an arm of this gate is the only way the gate can decide the PPL half of its own verdict.
run cand-prefill-ppl 2048 512 "$CHUNKS_PF" "$CAND"
echo
echo "--- verify  (-b 8 -ub 5, $CHUNKS_DEC chunks: the MTP width) ---"
run base-verify     8 5 "$CHUNKS_DEC" "$BASE" --save-all-logits "$WORK/kld-base-verify.gguf"
run cand-verify     8 5 "$CHUNKS_DEC" "$CAND" --kl-divergence --kl-divergence-base "$WORK/kld-base-verify.gguf"
run floor-verify    8 5 "$CHUNKS_DEC" "$BASE" --kl-divergence --kl-divergence-base "$WORK/kld-base-verify.gguf"
run cand-verify-ppl 8 5 "$CHUNKS_DEC" "$CAND"
echo
echo "--- decode  (-b 8 -ub 1, $CHUNKS_DEC chunks: single-token) ---"
run base-decode     8 1 "$CHUNKS_DEC" "$BASE" --save-all-logits "$WORK/kld-base-decode.gguf"
run cand-decode     8 1 "$CHUNKS_DEC" "$CAND" --kl-divergence --kl-divergence-base "$WORK/kld-base-decode.gguf"
run floor-decode    8 1 "$CHUNKS_DEC" "$BASE" --kl-divergence --kl-divergence-base "$WORK/kld-base-decode.gguf"
run cand-decode-ppl 8 1 "$CHUNKS_DEC" "$CAND"
echo
echo "floor-verify and floor-decode must read Mean KLD 0.000000 (or -0.000000) for the rest to mean anything."
echo "PPL is the ground truth: compare cand-*-ppl against the base-* row on the same shape; mean KLD is"
echo "judged against a quantization step (0.0306 for Q4_K_M vs Q8_0, 0.0103 for Q6_K_XL vs Q8_0)."
echo "per-chunk rows stay in $WORK/*.log"
