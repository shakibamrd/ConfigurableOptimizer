#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH -o logs/%x.%N.%j.out # STDOUT
#SBATCH -e logs/%x.%N.%j.err # STDERR
#SBATCH -a 0-3 # array size
#SBATCH --cpus-per-task=8 # Number of CPU cores per task
#SBATCH -J Motivation-DARTS-Baseline # sets the job name. If not
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CUPS_PER_NODE gpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

source ~/.bashrc
conda activate confopt

searchspace=darts
dataset=cifar10
sampler=darts
epochs=50
meta_info="Motivation-DARTS-DARTS-Freeze-ALl-But-Ops-Warm-Start-Baseline"

python scripts/experiments/run_experiment.py \
        --searchspace $searchspace \
        --dataset $dataset \
        --sampler $sampler \
        --epochs $epochs \
        --seed $SLURM_ARRAY_TASK_ID \
        --project-name iclr-experiments \
        --meta-info $meta_info \

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
