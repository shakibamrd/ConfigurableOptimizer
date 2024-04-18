#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080 #relea_gpu-rtx2080 #mlhiwidlc_gpu-rtx2080 # partition (queue)
#SBATCH -o logs/%x.%N.%j.out # STDOUT
#SBATCH -e logs/%x.%N.%j.err # STDERR
#SBATCH -a 1-3 # array size
#SBATCH -J LoRA-DARTS-DARTS-WE # sets the job name.
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CUPS_PER_NODE gpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <searchspace> <sampler> <weight-entanglement>"
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
if [ "$3" == "true" ]; then
    $entanglement="--entangle_op_weights"
elif [ "$3" == "false" ]; then
    $entanglement=""
else
    echo "Error: weight-entanglement must be 'true' or 'false'"
    exit 1
fi

searchspace=$1
sampler=$2

start=`date +%s`

source ~/.bashrc
conda activate confopt

python scripts/lora/run_we_experiment.py --sampler $sampler --wandb_log --space $searchspace --seed $SLURM_ARRAY_TASK_ID $entanglement

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
