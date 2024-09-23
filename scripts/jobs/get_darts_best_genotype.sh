#!/bin/bash

# add exps here
genotype_exps=("test")

for exp in "${genotype_exps[@]}"; do
    genotype_folder="scripts/genotypes/${exp}"
    bash scripts/jobs/submit_darts_best_genotype_jobs.sh $genotype_folder
done 
