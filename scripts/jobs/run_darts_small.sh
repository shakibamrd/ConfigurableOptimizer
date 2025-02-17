#!/bin/bash

searchspace="darts"
sampler="darts"
search_epochs=10
arch_lrs=(3e-4)


sbatch scripts/jobs/submit_darts_small.sh $searchspace $sampler $search_epochs 4 8
# sbatch scripts/jobs/submit_darts_small.sh $searchspace $sampler $search_epochs 8 4