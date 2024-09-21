#!/bin/bash

samplers=(darts drnas)
datasets=(cifar10 cifar100 imgnet16_120)
epochs=100
comments="'None'"

for sampler in "${samplers[@]}"; do
    for dataset in "${datasets[@]}"; do
        if [ "$sampler" = "darts" ]; then
            meta_info="'NB201-DARTS-Baseline'"
        elif [ "$sampler" = "drnas" ]; then
            meta_info="'NB201-DrNAS-Basic-Baseline'"
        else
            echo "invalid sampler"
            exit 1
        fi
        echo scripts/jobs/submit_nb201_baseline.sh $sampler $dataset $epochs 
        sbatch -J ${sampler}-${space}-${dataset}-epochs${epochs}-baseline scripts/jobs/submit_nb201_baseline.sh $dataset $sampler $epochs $meta_info $comments
    done
done

