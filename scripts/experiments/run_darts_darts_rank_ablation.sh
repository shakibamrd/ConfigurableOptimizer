#!/bin/bash

ranks=(1 2 4 8)

searchspace=darts
dataset=cifar10
sampler=darts
epochs=50
lora_warmup=16
meta_info="'DARTS LoRA Rank Ablation'"
comments="'None'"

for rank in "${ranks[@]}"; do
    exp_name=${searchspace}-${dataset}-${sampler}-epochs${epochs}-rank${rank}-warm${lora_warmup}
    echo $exp_name scripts/jobs/submit_lora_experiment_job.sh $searchspace $dataset $sampler $epochs $rank $lora_warmup $meta_info $comments
    sbatch -J $exp_name scripts/jobs/submit_lora_experiment_job.sh $searchspace $dataset $sampler $epochs $rank $lora_warmup $meta_info $comments
done

