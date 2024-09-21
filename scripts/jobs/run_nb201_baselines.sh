#!/bin/bash

samplers=(darts)
datasets=(cifar10 cifar100)
epochs=100
comments="'None'"

for sampler in "${samplers[@]}"; do
    for dataset in "${datasets[@]}"; do
        if [ "$sampler" = "darts" ]; then
            meta_info="NB201-DARTS-Baseline"
        elif [ "$sampler" = "drnas" ]; then
            meta_info="NB201-DrNAS-Baseline-Basic"
        else
            echo "invalid sampler"
            exit 1
        fi
        echo scripts/jobs/submit_nb201_baseline.sh $dataset $sampler $epochs $meta_info $comments
        sbatch -J ${sampler}-${space}-${dataset}-epochs${epochs}-baseline scripts/jobs/submit_nb201_baseline.sh $dataset $sampler $epochs $meta_info $comments
    done
done

