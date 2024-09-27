#!/bin/bash

# add genotype paths here
genotypes=("DARTS-Baseline/0")
dataset=cifar10
batch_size=48  # 96 / 2
epochs=600
comments="'None'"

for genotype in "${genotypes[@]}"; do
    exp_name="${genotype%%/*}"
    genotype_path="scripts/genotypes/${genotype}/genotype.txt"
    meta_info=${exp_name}-discrete
    echo scripts/jobs/submit_train_genotype.sh $exp_name $genotype_path $meta_info
    sbatch -J DARTS-Train-Genotype-${exp_name} scripts/jobs/submit_train_genotype.sh $genotype_path $dataset $batch_size $epochs $meta_info $comments
done

