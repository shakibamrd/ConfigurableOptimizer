#!/bin/bash

thresholds=(0.3 0.4 0.5)

searchspace=darts
dataset=cifar10
sampler=darts
epochs=50
frequency=20
meta_info="'DARTS-OLES-Threshold-Ablation'"
comments="'None'"

for threshold in "${thresholds[@]}"; do
    exp_name=${searchspace}-${dataset}-${sampler}-epochs${epochs}-threshold${threshold}
    echo $exp_name scripts/jobs/submit_oles_experiment_job.sh $searchspace $dataset $sampler $epochs $frequency $threshold $meta_info $comments
    sbatch -J $exp_name scripts/jobs/submit_oles_experiment_job.sh $searchspace $dataset $sampler $epochs $frequency $threshold $meta_info $comments
done

