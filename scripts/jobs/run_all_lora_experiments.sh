#!/bin/bash

spaces=("darts" "nb201")
samplers=("darts" "drnas" "gdas")

for space in "${spaces[@]}"; do
    for sampler in "${samplers[@]}"; do
        sbatch -J LoRA-${sampler}-${space}-WE scripts/jobs/submit_lora_experiment.sh $space $sampler
    done
done
