#!/bin/bash

searchspace="darts"

# write a list of genotypes to train
genotypes=(
  "DARTSGenotype(normal=[('sep_conv_5x5', 0), ('skip_connect', 1), ('dil_conv_3x3', 2), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_5x5', 3), ('skip_connect', 0), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('dil_conv_3x3', 2), ('sep_conv_5x5', 4)], reduce_concat=range(2, 6))"
  "DARTSGenotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))"
  "DARTSGenotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))"
  "DARTSGenotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))"
  "DARTSGenotype(normal=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 3), ('sep_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 2), ('sep_conv_5x5', 0), ('dil_conv_3x3', 0), ('skip_connect', 3), ('skip_connect', 3), ('skip_connect', 4)], reduce_concat=range(2, 6))"
  "DARTSGenotype(normal=[('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 2), ('skip_connect', 3), ('max_pool_3x3', 0), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))"
  "DARTSGenotype(normal=[('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('skip_connect', 2), ('dil_conv_5x5', 3), ('skip_connect', 2), ('sep_conv_3x3', 0)], reduce_concat=range(2, 6))"
  "DARTSGenotype(normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1), ('max_pool_3x3', 1), ('sep_conv_5x5', 0)], reduce_concat=range(2, 6))"
)

# change run name
run_names=(
    "darts_we_vanilla"
    "darts_ws_vanilla"
    "drnas_we_vanilla"
    "drnas_ws_vanilla"
    "darts_we_lora_rank_1"
    "darts_ws_lora_rank_1"
    "drnas_we_lora_rank_1"
    "drnas_ws_lora_rank_1"
)

if [ ${#genotypes[@]} -ne ${#run_names[@]} ]; then
  echo "Error: The lengths of the genotypes and run_names arrays do not match."
  exit 1
fi

# Iterate over the lists and call the existing script
for i in "${!genotypes[@]}"; do
  genotype=${genotypes[$i]}
  run_name=${run_names[$i]}
  echo $genotype
  sbatch scripts/jobs/train_discrete_genotype.sh "$searchspace" "$run_name" "$genotype" 
done