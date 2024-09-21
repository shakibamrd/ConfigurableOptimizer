#!/bin/bash

sampler=drnas
datasets=(cifar10 cifar100 imgnet16_120)
epochs=100
meta_info="'NB201-DrNAS-Prune-Baseline'"
comments="'None'"


for dataset in "${datasets[@]}"; do
    echo scripts/jobs/submit_nb201_baseline.sh $sampler $dataset $epochs
    sbatch -J ${sampler}-${space}-${dataset}-epochs${epochs}-prune-baseline scripts/jobs/submit_nb201_drnas_prune_baseline.sh $dataset $sampler $epochs $comments
done
