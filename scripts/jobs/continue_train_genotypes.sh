#!/bin/bash

# add genotype paths here
genotypes=("test_gen")
continue_paths=("scripts/blabla/test_discrete")

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
