#!/bin/bash
#SBATCH -p relea_gpu-rtx2080
#SBATCH -o logs/%x.%N.%j.out # STDOUT
#SBATCH -e logs/%x.%N.%j.err # STDERR
#SBATCH -a 0-3 # array size
#SBATCH --cpus-per-task=8 # Number of CPU cores per task
#SBATCH --time=0-1:30:00
#SBATCH -J Small-DARTS-Experiment # sets the job name. If not
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CUPS_PER_NODE gpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

source ~/.bashrc
conda activate my_confopt

searchspace=$1
sampler=$2
search_epochs=$3
num_channels=$4
layers=$5
arch_lr=$6


python scripts/experiments/darts_small_ss.py \
        --searchspace $searchspace \
        --sampler $sampler \
        --search_epochs $search_epochs \
        --num_channels $num_channels \
        --layers $layers \
        --seed $SLURM_ARRAY_TASK_ID \
        --arch_lr $arch_lr


end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
