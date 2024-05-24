#!/bin/bash

searchspace="darts"

# write a list of genotypes to train
genotypes=(
    "DARTSGenotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))"
    "DARTSGenotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))"
)

# change run name
run_names=(
    "run_1"
    "run_2"
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