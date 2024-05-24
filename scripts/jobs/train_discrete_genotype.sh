#!/bin/bash
#SBATCH -p ml_gpu-rtx2080 #relea_gpu-rtx2080 #mlhiwidlc_gpu-rtx2080 # partition (queue)
#SBATCH -o logs/%x.%N.%j.out # STDOUT
#SBATCH -e logs/%x.%N.%j.err # STDERR
#SBATCH -a 1-3 # array size
#SBATCH --time 3-00:00:00 # time (D-HH:MM)
#SBATCH --cpus-per-task 2
#SBATCH -J TRAIN_GENOTYPE # sets the job name.
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CUPS_PER_NODE gpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Check if the samplers are darts, drnas or gdas
searchspace=$1
run_name=$2
genotype=$3

start=`date +%s`

source ~/.bashrc
conda activate confopt

# export WANDB_MODE="offline"
python scripts/train_discrete/train_genotype.py --searchspace "$searchspace" --genotype "$genotype"  --run_name "$run_name"

echo Runtime: $runtime
