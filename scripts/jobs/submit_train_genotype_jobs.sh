#!/bin/bash

searchspace="darts"

run_names=(
  "darts_we_lora_epoch_50"
  "darts_ws_lora_epoch_50"
  "darts_we_vanilla_epoch_50"
  "darts_ws_vanilla_epoch_50"
)

for i in "${!run_names[@]}"; do
  run_name=${run_names[$i]}
  sbatch -J $run_name scripts/jobs/train_discrete_genotype.sh "$searchspace" "$run_name"
done
