#!/bin/bash

spaces=("darts" "nb201")
samplers=("darts" "drnas" "gdas")
we=("true" "false")
rank=4

for space in "${spaces[@]}"; do
    for sampler in "${samplers[@]}"; do
        for entanglement in "${we[@]}"; do
            echo scripts/jobs/submit_lora_experiment.sh $space $sampler $entanglement
            sbatch -J LoRA-r${rank}--${sampler}-${space}-WE-${entanglement} scripts/jobs/submit_lora_experiment.sh $space $sampler $entanglement $rank
        done
    done
done

