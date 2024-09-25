#!/bin/bash

spaces=(S1 S2 S3)
datasets=(cifar10)
samplers=(darts drnas)
epochs=100
rank=1
lora_warmup_epochs=16
comments="'None'"

for sampler in "${samplers[@]}"; do
    for dataset in "${datasets[@]}"; do
        for space in "${spaces[@]}"; do
            exp_name=nb1shot1-${space}-${dataset}-${sampler}-epochs${epochs}-rank${rank}-warm${lora_warmup_epochs}
            if [ "$sampler" = "darts" ]; then
                meta_info="NB1Shot1-DARTS-LoRA"
            elif [ "$sampler" = "drnas" ]; then
                meta_info="NB1Shot1-DrNAS-LoRA"
            else
                echo "invalid sampler"
                exit 1
            fi
            echo $exp_name scripts/jobs/submit_nb1shot1_lora.sh $space $dataset $sampler $epochs $rank $lora_warmup_epochs $meta_info $comments
            sbatch -J $exp_name scripts/jobs/submit_nb1shot1_lora.sh $space $dataset $sampler $epochs $rank $lora_warmup_epochs $meta_info $comments
        done
    done
done
