#!/bin/bash

# add exps here
genotype_exps=("DARTS-Baseline" "DARTS-LoRA-Rank-Ablation" "DARTS-OLES-Threshold-Ablation" "DrNAS-Baseline" "DrNAS-Baseline-Basic" "DrNAS-LoRA-Rank-Ablation" "DrNAS-LoRA-Rank-Ablation-Basic" "DrNAS-OLES-Threshold-Ablation" "DrNAS-OLES-Threshold-Ablation-Basic")

for exp in "${genotype_exps[@]}"; do
    genotype_folder="scripts/genotypes/${exp}"
    sbatch -J DARTS-Genotype-100-epochs-${exp} scripts/jobs/submit_darts_best_genotype_jobs.sh $genotype_folder
done 
