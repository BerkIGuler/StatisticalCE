#!/bin/bash

# Generate 3x3 NMSE vs SNR figures for all configurations
# 5 delay profiles × 3 pilot types = 15 figures

DELAY_PROFILES=("A" "B" "C" "D" "E")
# Pilot type suffixes used in filenames: '2', '23', '2711'
PILOT_TYPES=("2" "23" "2711")

# Base results directory (where 'ind_dist' subfolder lives)
RESULTS_DIR="${RESULTS_DIR:-results}"

TOTAL=$(( ${#DELAY_PROFILES[@]} * ${#PILOT_TYPES[@]} ))
RUN=0

for profile in "${DELAY_PROFILES[@]}"; do
  for pilot in "${PILOT_TYPES[@]}"; do
      RUN=$((RUN + 1))
      echo "===================================================="
      echo "Figure ${RUN}/${TOTAL} | TDL-${profile} | pilots=${pilot}"
      echo "===================================================="

      OUT_PATH="${RESULTS_DIR}/figs/ind_dist_TDL${profile}_pilots${pilot}.png"

      python plot_ind_dist_results.py \
        --delay_profile "${profile}" \
        --pilot_type "${pilot}" \
        --results_dir "${RESULTS_DIR}" \
        --output_path "${OUT_PATH}"
  done
done

echo "Done. All ${TOTAL} figures written under ${RESULTS_DIR}/figs/"
