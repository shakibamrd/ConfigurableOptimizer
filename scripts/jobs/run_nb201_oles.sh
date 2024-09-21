#!/bin/bash

searchspace=nb201
datasets=(cifar10 cifar100 imgnet16_120)
samplers=(darts drnas)
epochs=100
threshold=0.7
frequency=20
meta_info="'DrNAS-OLES-Threshold-Ablation'"
comments="'None'"

for sampler in "${samplers[@]}"; do
    for dataset in "${datasets[@]}"; do
        if [ "$sampler" = "darts" ]; then
            meta_info="'NB201-DARTS-OLES'"
        elif [ "$sampler" = "drnas" ]; then
            meta_info="'NB201-DrNAS-OLES'"
        else   
            echo $sampler
            echo "invalid sampler"
            exit 1
        fi
        exp_name=${searchspace}-${dataset}-${sampler}-oles-epochs${epochs}-threshold${threshold}
        echo $exp_name scripts/jobs/submit_oles_experiment_job.sh $searchspace $dataset $sampler $epochs $frequency $threshold $meta_info $comments
        # sbatch -J $exp_name scripts/jobs/submit_oles_experiment_job.sh $searchspace $dataset $sampler $epochs $frequency $threshold $meta_info $comments
        bash scripts/jobs/submit_oles_experiment_job.sh $searchspace $dataset $sampler $epochs $frequency $threshold $meta_info $comments
    done
done

