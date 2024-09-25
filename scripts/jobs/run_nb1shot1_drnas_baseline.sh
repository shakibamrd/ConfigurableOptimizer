#!/bin/bash

dataset=cifar10
nb1shot1_spaces=(S1 S2 S3)
epochs=50
comments="'None'"


for space in "${nb1shot1_spaces[@]}"; do
    echo scripts/jobs/submit_nb1shot1_drnas_prune_baseline.sh $space $dataset $epochs $comments
    sbatch -J drnas-nb1shot1-${space}-${dataset}-epochs${epochs}-prune-baseline scripts/jobs/submit_nb1shot1_drnas_prune_baseline.sh $space $dataset $epochs $comments
done

