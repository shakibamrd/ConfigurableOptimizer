#!/bin/bash



spaces=(S1 S2 S3)
dataset=cifar10
samplers=(darts drnas)
epochs=100
frequency=20
thresholds=(0.2 0.3 0.4 0.5 0.6 0.7)

comments="'None'"

for sampler in "${samplers[@]}"; do
    if [ "$sampler" = "darts" ]; then
        meta_info="NB1Shot1-DARTS-OLES"
    elif [ "$sampler" = "drnas" ]; then
        meta_info="NB1Shot1-DrNAS-OLES"
    else
        echo "invalid sampler"
        exit 1
    fi
    for space in "${spaces[@]}"; do
        for threshold in "${thresholds[@]}"; do
            exp_name=nb1shot1-${space}-${dataset}-${sampler}-epochs${epochs}-threshold${threshold}
            echo $exp_name scripts/jobs/submit_nb1shot1_oles.sh $space $dataset $sampler $epochs $frequency $threshold $meta_info $comments
            sbatch -J $exp_name scripts/jobs/submit_nb1shot1_oles.sh $space $dataset $sampler $epochs $frequency $threshold $meta_info $comments
        done
    done
done

