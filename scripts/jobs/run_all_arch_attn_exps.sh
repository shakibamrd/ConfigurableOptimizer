#!/bin/bash

spaces=("darts" "nb201")
samplers=("darts" "drnas" "gdas" "reinmax")

for space in "${spaces[@]}"; do
    for sampler in "${samplers[@]}"; do
        if [ $sampler == "darts" ]; then
            epochs=50
        elif [ $sampler == "drnas" ]; then
            epochs=100
        elif [ $sampler == "gdas" ]; then
            epochs=250
        elif [ $sampler == "reinmax" ]; then
            epochs=250
        else
            echo "Unknown sampler"
            exit 1
        fi

        echo scripts/jobs/submit_arch_attn_exp.sh $space $sampler $epochs
        sbatch -J ${sampler}-${space}-k${k}-warm_epochs${warm_epochs} scripts/jobs/submit_arch_attn_partial_connection_exp.sh $space $sampler $epochs

    done
done

