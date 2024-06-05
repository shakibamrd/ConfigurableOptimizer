#!/bin/bash

searchspace="darts"

# write a list of genotypes to train
# change run name
# run_names=(
#     "darts_we_vanilla"
#     "darts_ws_vanilla"
#     "drnas_we_vanilla"
#     "drnas_ws_vanilla"
#     "darts_we_lora_rank_1"
#     "darts_ws_lora_rank_1"
#     "drnas_we_lora_rank_1"
#     "drnas_ws_lora_rank_1"
# )

# run_names=(
  # "darts_we_vanilla"
  # "darts_ws_vanilla"
  # "darts_we_lora_rank_1"
  # "darts_ws_lora_rank_1"
  # "darts_we_lora_alt"
  # "darts_ws_lora_alt"
  # "drnas_we_vanilla"
  # "drnas_ws_vanilla"
  # "drnas_we_lora_rank_1"
  # "drnas_ws_lora_rank_1"
  # "drnas_we_lora_alt"
  # "drnas_ws_lora_alt"
# )

run_names=(
 "darts_we_lora_epoch_50"
 "darts_ws_lora_epoch_50"
 "darts_we_vanilla_epoch_50"
 "darts_ws_vanilla_epoch_50"
)

# turn this to true for continuing run and false for a fresh run
load_saved_model="true"

# Iterate over the lists and call the existing script
for i in "${!run_names[@]}"; do
  run_name=${run_names[$i]}
  sbatch -J $run_name scripts/jobs/train_discrete_genotype.sh "$searchspace" "$run_name" "$load_saved_model"
done
