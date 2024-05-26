#!/bin/bash

spaces=("darts" "nb201")
samplers=("darts" "drnas")
we=("false" "true")
rank=0

for space in "${spaces[@]}"; do
    for sampler in "${samplers[@]}"; do
        for entanglement in "${we[@]}"; do
            echo scripts/jobs/submit_lora_experiment.sh $space $sampler $entanglement
            sbatch -J LoRA-${sampler}-${space}-WE-${entanglement}-300epochs scripts/jobs/submit_lora_experiment.sh $space $sampler $entanglement $rank
        done
    done
done

