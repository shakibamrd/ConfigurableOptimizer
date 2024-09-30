#!/bin/bash

# add genotype paths here
genotypes=(DrNAS-LoRA-Rank-Ablation DrNAS-LoRA-Rank-Ablation DrNAS-LoRA-Rank-Ablation DrNAS-OLES-Threshold-Ablation-Basic DrNAS-OLES-Threshold-Ablation-Basic DrNAS-OLES-Threshold-Ablation-Basic DrNAS-OLES-Threshold-Ablation DrNAS-OLES-Threshold-Ablation DrNAS-OLES-Threshold-Ablation DrNAS-OLES-Threshold-Ablation DARTS-Prune-LoRA-Warmup-Ablation DrNAS-Baseline DrNAS-Baseline DARTS-Prune-LoRA-Warmup-Ablation DrNAS-Baseline DrNAS-Baseline DARTS-Prune-LoRA-Warmup-Ablation DARTS-Prune-LoRA-Warmup-Ablation DARTS-Prune-Baseline DARTS-LoRA-Rank-Ablation DARTS-LoRA-Rank-Ablation DARTS-OLES-Threshold-Ablation DARTS-Prune-Baseline DARTS-LoRA-Rank-Ablation DARTS-Baseline DARTS-OLES-Threshold-Ablation DARTS-OLES-Threshold-Ablation DARTS-OLES-Threshold-Ablation DARTS-LoRA-Rank-Ablation DARTS-Baseline DARTS-Prune-Baseline DARTS-Baseline)
continue_paths=(logs/DISCRETE_darts-cifar10_seed1/darts/cifar10/1/discrete/2024-09-29-09:50:44.161 logs/DISCRETE_darts-cifar10_seed2/darts/cifar10/2/discrete/2024-09-29-09:50:44.161 logs/DISCRETE_darts-cifar10_seed0/darts/cifar10/0/discrete/2024-09-29-09:50:44.161 logs/DISCRETE_darts-cifar10_seed2/darts/cifar10/2/discrete/2024-09-29-09:50:44.051 logs/DISCRETE_darts-cifar10_seed0/darts/cifar10/0/discrete/2024-09-29-09:50:44.049 logs/DISCRETE_darts-cifar10_seed1/darts/cifar10/1/discrete/2024-09-29-09:50:44.050 logs/DISCRETE_darts-cifar10_seed3/darts/cifar10/3/discrete/2024-09-29-09:50:30.584 logs/DISCRETE_darts-cifar10_seed0/darts/cifar10/0/discrete/2024-09-29-09:50:30.585 logs/DISCRETE_darts-cifar10_seed1/darts/cifar10/1/discrete/2024-09-29-09:50:30.585 logs/DISCRETE_darts-cifar10_seed2/darts/cifar10/2/discrete/2024-09-29-09:50:30.584 logs/DISCRETE_darts-cifar10_seed1/darts/cifar10/1/discrete/2024-09-29-09:50:30.584 logs/DISCRETE_darts-cifar10_seed1/darts/cifar10/1/discrete/2024-09-29-09:50:30.585 logs/DISCRETE_darts-cifar10_seed2/darts/cifar10/2/discrete/2024-09-29-09:50:30.584 logs/DISCRETE_darts-cifar10_seed2/darts/cifar10/2/discrete/2024-09-29-09:50:30.585 logs/DISCRETE_darts-cifar10_seed3/darts/cifar10/3/discrete/2024-09-29-09:50:30.585 logs/DISCRETE_darts-cifar10_seed0/darts/cifar10/0/discrete/2024-09-29-09:50:30.585 logs/DISCRETE_darts-cifar10_seed0/darts/cifar10/0/discrete/2024-09-29-09:50:30.585 logs/DISCRETE_darts-cifar10_seed3/darts/cifar10/3/discrete/2024-09-29-09:50:30.584 logs/DISCRETE_darts-cifar10_seed1/darts/cifar10/1/discrete/2024-09-29-09:47:53.339 logs/DISCRETE_darts-cifar10_seed2/darts/cifar10/2/discrete/2024-09-29-09:47:53.340 logs/DISCRETE_darts-cifar10_seed1/darts/cifar10/1/discrete/2024-09-29-09:47:53.335 logs/DISCRETE_darts-cifar10_seed0/darts/cifar10/0/discrete/2024-09-29-09:47:53.332 logs/DISCRETE_darts-cifar10_seed3/darts/cifar10/3/discrete/2024-09-29-09:47:53.333 logs/DISCRETE_darts-cifar10_seed3/darts/cifar10/3/discrete/2024-09-29-09:47:53.333 logs/DISCRETE_darts-cifar10_seed3/darts/cifar10/3/discrete/2024-09-29-09:47:53.333 logs/DISCRETE_darts-cifar10_seed3/darts/cifar10/3/discrete/2024-09-29-09:47:53.333 logs/DISCRETE_darts-cifar10_seed2/darts/cifar10/2/discrete/2024-09-29-09:47:53.339 logs/DISCRETE_darts-cifar10_seed1/darts/cifar10/1/discrete/2024-09-29-09:47:53.332 logs/DISCRETE_darts-cifar10_seed0/darts/cifar10/0/discrete/2024-09-29-09:47:53.336 logs/DISCRETE_darts-cifar10_seed2/darts/cifar10/2/discrete/2024-09-29-09:47:53.356 logs/DISCRETE_darts-cifar10_seed2/darts/cifar10/2/discrete/2024-09-29-09:47:53.333 logs/DISCRETE_darts-cifar10_seed0/darts/cifar10/0/discrete/2024-09-29-09:47:05.298)

if [ "${#genotypes[@]}" -ne "${#continue_paths[@]}" ]; then
    echo "Both genotypes and continue_folders should contain same number of paths"
    exit 1
fi

dataset=cifar10
batch_size=96
epochs=600
comments="train-best-from-supernet-continue"

length="${#genotypes[@]}"

for ((i=0; i<length; i++)); do
    genotype=${genotypes[$i]}
    continue_path=${continue_paths[$i]}

    exp_name="${genotype%%/*}"
    genotype_path="scripts/retrain/genotypes/${genotype}/best_genotype.txt"
    meta_info=${exp_name}-continue-discrete

    continue_runtime="$(basename "$continue_path")"
    continue_folder="$(dirname "$continue_path")"
    echo sbatch -J Train-Genotype-${exp_name} scripts/jobs/submit_train_genotype_continue.sh $genotype_path $continue_runtime $continue_folder $dataset $batch_size $epochs $meta_info $comments
    sbatch -J Train-Genotype-continue-${exp_name} scripts/jobs/submit_train_genotype_continue.sh $genotype_path $continue_folder $continue_runtime $dataset $batch_size $epochs $meta_info $comments
done
