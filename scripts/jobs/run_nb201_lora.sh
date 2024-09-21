#!/bin/bash

searchspace=nb201
datasets=(cifar100 cifar10 imgnet16_120)
samplers=(darts drnas)
epochs=100
rank=1
lora_warmup_epochs=16
meta_info="'NB201-DARTS-LoRA'"
comments="'None'"

for sampler in "${samplers[@]}"; do
    for dataset in "${datasets[@]}"; do
        exp_name=${searchspace}-${dataset}-${sampler}-epochs${epochs}-rank${rank}-warm${lora_warmup_epochs}
        if [ "$sampler" = "darts" ]; then
            meta_info="'NB201-DARTS-LoRA'"
        elif [ "$sampler" = "drnas" ]; then
            meta_info="'NB201-DrNAS-LoRA'"
        else
            echo "invalid sampler"
            exit 1
        fi
        echo $exp_name scripts/jobs/submit_lora_experiment_job.sh $searchspace $dataset $sampler $epochs $rank $lora_warmup_epochs $meta_info $comments
        sbatch -J $exp_name scripts/jobs/submit_lora_experiment_job.sh $searchspace $dataset $sampler $epochs $rank $lora_warmup_epochs $meta_info $comments
    done
done