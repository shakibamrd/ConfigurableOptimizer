#!/bin/bash

spaces=("darts")
samplers=("darts" "drnas" "gdas")
t0s=(10 20 25)
tmults=(1 2 4)

for space in "${spaces[@]}"; do
    for sampler in "${samplers[@]}"; do
        for t0 in "${t0s[@]}"; do
            for tmult in "${tmults[@]}"; do
                # echo scripts/jobs/submit_cosine_annealing_experiment.sh $space cifar10 50 $sampler $t0 $tmult
                echo sbatch -J $space-$sampler-t0$t0-tmult$tmult-cf10-50e scripts/jobs/submit_cosine_annealing_experiment.sh $space cifar10 50 $sampler $t0 $tmult
                python scripts/baselines/run_search.py --sampler $sampler --searchspace $space --dataset cifar10 --search_epochs 50 --cosine_anneal_restarts_T0 $t0 --cosine_anneal_restarts_Tmult $tmult --seed 9001

            done
        done
    done
done
