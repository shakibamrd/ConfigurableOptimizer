#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH -o logs/%x.%N.%j.out # STDOUT
#SBATCH -e logs/%x.%N.%j.err # STDERR
#SBATCH -a 0-3 # array size
#SBATCH --cpus-per-task=8 # Number of CPU cores per task
#SBATCH -J Motivation-NB201-LoRA # sets the job name. If not
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CUPS_PER_NODE gpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

source ~/.bashrc
conda activate confopt

searchspace=nb201
dataset=cifar10
sampler=darts
epochs=100
lora_rank=1
lora_warmup=16
meta_info="Motivation-NB201-DARTS-Freeze-ALl-But-Ops-LoRA"

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

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
