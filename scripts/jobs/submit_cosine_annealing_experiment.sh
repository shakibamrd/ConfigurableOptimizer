#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080 #relea_gpu-rtx2080 #mlhiwidlc_gpu-rtx2080 # partition (queue)
#SBATCH -o logs/%x.%N.%j.out # STDOUT
#SBATCH -e logs/%x.%N.%j.err # STDERR
#SBATCH -a 1-3 # array size
#SBATCH --cpus-per-task 8
#SBATCH -J DARTS-Cosine-Warm-Restarts # sets the job name.
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CUPS_PER_NODE gpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <searchspace> <dataset> <search_epochs> <sampler> <cosine_anneal_restarts_T0> <cosine_anneal_restarts_Tmult>"
    exit 1
fi


searchspace=$1
dataset=$2
search_epochs=$3
sampler=$4
cosine_anneal_restarts_T0=$5
cosine_anneal_restarts_Tmult=$6


# Check if the samplers are darts, drnas or gdas
if [ "$sampler" != "darts" ] && [ "$sampler" != "drnas" ] && [ "$sampler" != "gdas" ]; then
    echo "Error: optimizer must be 'darts', 'drnas' or 'gdas'"
    exit 1
fi

start=`date +%s`

source ~/.bashrc
conda activate confopt

export WANDB_MODE="offline"

python scripts/baselines/run_search.py --sampler $sampler --searchspace $searchspace --dataset $dataset --search_epochs $search_epochs --cosine_anneal_restarts_T0 $cosine_anneal_restarts_T0 --cosine_anneal_restarts_Tmult $cosine_anneal_restarts_Tmult --seed $SLURM_ARRAY_TASK_ID

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
