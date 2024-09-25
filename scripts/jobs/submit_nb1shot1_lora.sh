#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH -o logs/%x.%N.%j.out # STDOUT
#SBATCH -e logs/%x.%N.%j.err # STDERR
#SBATCH -a 0-3 # array size
#SBATCH --cpus-per-task=8 # Number of CPU cores per task
#SBATCH -J NB1Shot1-DARTS-Experiment # sets the job name. If not
#SBATCH --exclude=dlcgpu05
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CUPS_PER_NODE gpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

source ~/.bashrc
conda activate confopt

searchspace=nb1shot1
nb1shot1_searchspace=$1
dataset=$2
sampler=$3
epochs=$4
lora_rank=$5
lora_warmup=$6
meta_info=$7
comments=$8
is_basic=$9

basic=""
if [ "$is_basic" = "1" ]; then
    basic="--drnas-basic"
fi

python scripts/experiments/run_experiment.py \
        --searchspace $searchspace \
        --nb1shot1-searchspace $nb1shot1_searchspace \
        --dataset $dataset \
        --sampler $sampler \
        --epochs $epochs \
        --lora-rank $lora_rank \
        --lora-warm-epochs $lora_warmup \
        --seed $SLURM_ARRAY_TASK_ID \
        --project-name iclr-experiments \
        --meta-info $meta_info \
        --comments $comments \
        $basic

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
