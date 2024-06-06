#!/bin/bash

searchspace="darts"

run_names=(
  "darts_we_lora_rank_1"
  "darts_ws_lora_rank_1"
  "darts_we_vanilla"
  "darts_ws_vanilla"
)

for i in "${!run_names[@]}"; do
  run_name=${run_names[$i]}
  sbatch -J $run_name scripts/jobs/train_discrete_genotype.sh "$searchspace" "$run_name"
done
