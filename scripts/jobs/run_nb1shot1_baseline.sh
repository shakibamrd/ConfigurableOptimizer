#!/bin/bash

samplers=(darts drnas)
dataset=cifar10
nb1shot1_spaces=(S1 S2 S3)
epochs=50
comments="'None'"

for sampler in "${samplers[@]}"; do
    for space in "${nb1shot1_spaces[@]}"; do
        if [ "$sampler" = "darts" ]; then
            meta_info="NB1Shot1-DARTS-Baseline"
        elif [ "$sampler" = "drnas" ]; then
            meta_info="NB1Shot1-DrNAS-Baseline-Basic"
        else
            echo "invalid sampler"
            exit 1
        fi
        echo scripts/jobs/submit_nb1shot1_baseline.sh $sampler $space $dataset $epochs $meta_info $comments
        sbatch -J ${sampler}-nb1shot1-${space}-${dataset}-epochs${epochs}-baseline scripts/jobs/submit_nb1shot1_baseline.sh $space $dataset $sampler $epochs $meta_info $comments
    done
done

