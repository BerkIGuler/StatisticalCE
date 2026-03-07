#!/bin/bash

# Bilinear interpolation evaluation: 5 profiles × 3 pilot configs = 15 runs

DELAY_PROFILES=("A" "B" "C" "D" "E")
# Pilot symbol options: [2], [2,3], [2,7,11]
PILOT_OPTS=("2" "2 3" "2 7 11")
SAVE_DIR="${SAVE_DIR:-results}"
EVAL_SNRs="0 5 10 15 20 25 30"

TOTAL=$(( ${#DELAY_PROFILES[@]} * ${#PILOT_OPTS[@]} ))
RUN=0

for profile in "${DELAY_PROFILES[@]}"; do
  for pilot in "${PILOT_OPTS[@]}"; do
    RUN=$((RUN + 1))
    echo "=============================================="
    echo "Run ${RUN}/${TOTAL} | TDL-${profile} | pilots [${pilot}]"
    echo "=============================================="
    python evaluate_bilinear_interp.py \
      --delay_profile "$profile" \
      --eval_SNRs $EVAL_SNRs \
      --pilot_symbols $pilot \
      --save_dir "$SAVE_DIR"
  done
done

echo "Done. All ${TOTAL} runs finished. Results in ${SAVE_DIR}/"
