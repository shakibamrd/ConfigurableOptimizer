#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080 #relea_gpu-rtx2080 #mlhiwidlc_gpu-rtx2080 # partition (queue)
#SBATCH -o logs/%x.%N.%j.out # STDOUT
#SBATCH -e logs/%x.%N.%j.err # STDERR
#SBATCH -a 1-3 # array size
#SBATCH --cpus-per-task 8
#SBATCH -J LoRA-DARTS-DARTS-WE # sets the job name.
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CUPS_PER_NODE gpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <searchspace> <sampler> <weight-entanglement> <rank>"
    exit 1
fi

# Check if the searchspace is either "darts" or "nb201"
if [ "$1" != "darts" ] && [ "$1" != "nb201" ]; then
    echo "Error: searchspace must be either 'darts' or 'nb201'"
    exit 1
fi

# Check if the samplers are darts, drnas or gdas
if [ "$2" != "darts" ] && [ "$2" != "drnas" ] && [ "$2" != "gdas" ]; then
    echo "Error: optimizer must be 'darts', 'drnas' or 'gdas'"
    exit 1
fi


# Check if the samplers are darts, drnas or gdas
searchspace=$1
sampler=$2
rank=$4

start=`date +%s`

source ~/.bashrc
conda activate confopt

# export WANDB_MODE="offline"

if [ "$3" == "true" ]; then
    python scripts/lora/run_we_experiment.py --use_lora --lora_warm_epoch 10 --lora_rank $rank --lora_alpha $rank --lora_merge_weights --sampler $sampler --wandb_log --searchspace $searchspace --entangle_op_weights --seed $SLURM_ARRAY_TASK_ID
elif [ "$3" == "false" ]; then
    python scripts/lora/run_we_experiment.py --use_lora --lora_warm_epoch 10 --lora_rank $rank --lora_alpha $rank --lora_merge_weights --sampler $sampler --wandb_log --searchspace $searchspace --seed $SLURM_ARRAY_TASK_ID
else
    echo "Error: weight-entanglement must be 'true' or 'false'"
    exit 1
fi

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
