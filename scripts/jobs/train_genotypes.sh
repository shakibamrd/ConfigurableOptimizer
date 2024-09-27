#!/bin/bash

# add genotype paths here
# genotypes=("DARTS-Baseline/0")
genotypes=("DARTS-Baseline"  "DARTS-LoRA-Rank-Ablation"  "DARTS-OLES-Threshold-Ablation"  "DARTS-Prune-Baseline"  "DARTS-Prune-LoRA-Warmup-Ablation"  "DrNAS-Baseline"  "DrNAS-Baseline-Basic"  "DrNAS-LoRA-Rank-Ablation"  "DrNAS-LoRA-Rank-Ablation-Basic"  "DrNAS-OLES-Threshold-Ablation"  "DrNAS-OLES-Threshold-Ablation-Basic")
dataset=cifar10
batch_size=96
epochs=600
comments="train-best-from-supernet"

for genotype in "${genotypes[@]}"; do
    exp_name="${genotype%%/*}"
    genotype_path="scripts/retrain/genotypes/${genotype}/best_genotype.txt"
    meta_info=${exp_name}-discrete

    echo sbatch -J Train-Genotype-${exp_name} scripts/jobs/submit_train_genotype.sh $genotype_path $dataset $batch_size $epochs $meta_info $comments
    sbatch -J Train-Genotype-${exp_name} scripts/jobs/submit_train_genotype.sh $genotype_path $dataset $batch_size $epochs $meta_info $comments
done

