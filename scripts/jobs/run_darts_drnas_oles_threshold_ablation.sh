#!/bin/bash

thresholds=(0.2 0.3 0.4 0.5)

searchspace=darts
dataset=cifar10
sampler=drnas
epochs=100
frequency=20
meta_info="'DrNAS-OLES-Threshold-Ablation-Basic'"
comments="'None'"

for threshold in "${thresholds[@]}"; do
    exp_name=${searchspace}-${dataset}-${sampler}-epochs${epochs}-threshold${threshold}
    echo $exp_name scripts/jobs/submit_oles_experiment_job.sh $searchspace $dataset $sampler $epochs $frequency $threshold $meta_info $comments
    sbatch -J $exp_name scripts/jobs/submit_oles_experiment_job.sh $searchspace $dataset $sampler $epochs $frequency $threshold $meta_info $comments
done

