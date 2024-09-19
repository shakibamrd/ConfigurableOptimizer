#!/bin/bash

ranks=(1 2 4 8)

searchspace=darts
dataset=cifar10
sampler=drnas
epochs=100
lora_warmup=16
meta_info="'DrNAS-LoRA-Rank-Ablation'"
comments="'None'"

for rank in "${ranks[@]}"; do
    exp_name=${searchspace}-${dataset}-${sampler}-epochs${epochs}-rank${rank}-warm${lora_warmup}
    echo $exp_name scripts/jobs/submit_lora_experiment_job.sh $searchspace $dataset $sampler $epochs $rank $lora_warmup $meta_info $comments
    sbatch -J $exp_name scripts/jobs/submit_lora_experiment_job.sh $searchspace $dataset $sampler $epochs $rank $lora_warmup $meta_info $comments
done

