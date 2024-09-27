#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH -o logs/%x.%N.%j.out # STDOUT
#SBATCH -e logs/%x.%N.%j.err # STDERR
#SBATCH -a 0-3 # array size
#SBATCH --cpus-per-task=8 # Number of CPU cores per task
#SBATCH --gres=gpu:2 # Number of GPU per task
#SBATCH -J DARTS-Genotype # sets the job name. If not
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CUPS_PER_NODE gpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

source ~/.bashrc
conda activate confopt


genotype_file=$1
dataset=$2
batch_size=$3
epochs=$4
meta_info=$5
comments=$6

if [ ! -e "$genotype_file" ]; then
  echo "Genotype file does not exist."
  exit 1
fi

genotype=$(<"$genotype_file")

torchrun scripts/experiments/train_darts_genotype.py \
        --genotype "$genotype" \
        --dataset $dataset \
        --seed  $SLURM_ARRAY_TASK_ID\
        --batch-size $batch_size \
        --epochs $epochs \
        --project-name iclr-train-genotypes \
        --meta-info $meta_info \
        --comments $comments \
        --lr 0.05


end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
