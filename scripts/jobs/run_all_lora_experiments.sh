#!/bin/bash

spaces=("darts")
samplers=("darts")
we=("false", "true")
rank=1

for space in "${spaces[@]}"; do
    for sampler in "${samplers[@]}"; do
        for entanglement in "${we[@]}"; do
            echo scripts/jobs/submit_lora_experiment.sh $space $sampler $entanglement
            sbatch -J LoRA-r${rank}-${sampler}-${space}-WE-${entanglement}-300epochs scripts/jobs/submit_lora_experiment.sh $space $sampler $entanglement $rank
        done
    done
done

