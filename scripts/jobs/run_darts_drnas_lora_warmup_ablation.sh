#!/bin/bash

lora_warmup_epochs=(2 4 8 16 32)

searchspace=darts
dataset=cifar10
sampler=drnas
epochs=100
rank=1
meta_info="'DrNAS-LoRA-Warmup-Ablation'"
comments="'None'"

for lora_warmup in "${lora_warmup_epochs[@]}"; do
    exp_name=${searchspace}-${dataset}-${sampler}-epochs${epochs}-rank${rank}-warm${lora_warmup}
    echo $exp_name scripts/jobs/submit_lora_experiment_job.sh $searchspace $dataset $sampler $epochs $rank $lora_warmup $meta_info $comments
    sbatch -J $exp_name scripts/jobs/submit_lora_experiment_job.sh $searchspace $dataset $sampler $epochs $rank $lora_warmup $meta_info $comments
done

