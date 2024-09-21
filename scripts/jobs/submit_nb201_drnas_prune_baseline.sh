#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH -o logs/%x.%N.%j.out # STDOUT
#SBATCH -e logs/%x.%N.%j.err # STDERR
#SBATCH -a 0-3 # array size
#SBATCH --cpus-per-task=8 # Number of CPU cores per task
#SBATCH -J DrNAS-Prune-Baseline # sets the job name. If not
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CUPS_PER_NODE gpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

source ~/.bashrc
conda activate confopt

searchspace=nb201
dataset=$1
sampler=$2
epochs=$3
meta_info="'NB201-DrNAS-Prune-Baseline'"
comments=$4
partial_connection_k=4
partial_connection_warm_epochs=15

python scripts/experiments/run_experiment.py \
        --searchspace $searchspace \
        --dataset $dataset \
        --sampler $sampler \
        --epochs $epochs \
        --seed $SLURM_ARRAY_TASK_ID \
        --project-name iclr-experiments \
        --meta-info $meta_info \
        --comments $comments \
        --prune \
        --partial-connection \
        --partial-connection-k $partial_connection_k \
        --partial-connection-warm-epochs $partial_connection_warm_epochs \

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
