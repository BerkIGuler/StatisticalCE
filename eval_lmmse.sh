#!/bin/bash

# Comprehensive LMMSE evaluation: 5 profiles × 3 pilot configs × 2 stats modes = 30 runs

DELAY_PROFILES=("A" "B" "C" "D" "E")
# Pilot symbol options: [2], [2,3], [2,7,11]
PILOT_OPTS=("2" "2 3" "2 7 11")
LMMSE_STATS_OPTS=("all_test")
SAVE_DIR="${SAVE_DIR:-results}"
EVAL_SNRs="0 5 10 15 20 25 30"

TOTAL=$(( ${#DELAY_PROFILES[@]} * ${#PILOT_OPTS[@]} * ${#LMMSE_STATS_OPTS[@]} ))
RUN=0

for profile in "${DELAY_PROFILES[@]}"; do
  for pilot in "${PILOT_OPTS[@]}"; do
    for stats in "${LMMSE_STATS_OPTS[@]}"; do
      RUN=$((RUN + 1))
      echo "=============================================="
      echo "Run ${RUN}/${TOTAL} | TDL-${profile} | pilots [${pilot}] | stats=${stats}"
      echo "=============================================="
      python evaluate_lmmse.py \
        --delay_profile "$profile" \
        --eval_SNRs $EVAL_SNRs \
        --pilot_symbols $pilot \
        --lmmse_stats "$stats" \
        --save_dir "$SAVE_DIR"
    done
  done
done

echo "Done. All ${TOTAL} runs finished. Results in ${SAVE_DIR}/"
