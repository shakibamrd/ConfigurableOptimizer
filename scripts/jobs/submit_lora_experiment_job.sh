#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH -o logs/%x.%N.%j.out # STDOUT
#SBATCH -e logs/%x.%N.%j.err # STDERR
#SBATCH -a 0-3 # array size
#SBATCH --cpus-per-task=8 # Number of CPU cores per task
#SBATCH -J LoRA-DARTS-Experiment # sets the job name. If not
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CUPS_PER_NODE gpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

source ~/.bashrc
conda activate confopt

searchspace=$1
dataset=$2
sampler=$3
epochs=$4
lora_rank=$5
lora_warmup=$6
meta_info=$7
comments=$8

python scripts/experiments/run_experiment.py \
        --searchspace $searchspace \
        --dataset $dataset \
        --sampler $sampler \
        --epochs $epochs \
        --lora-rank $lora_rank \
        --lora-warm-epochs $lora_warmup \
        --seed $SLURM_ARRAY_TASK_ID \
        --project-name iclr-experiments \
        --meta-info $meta_info \
        --comments $comments

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
