#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH -o logs/%x.%N.%j.out # STDOUT
#SBATCH -e logs/%x.%N.%j.err # STDERR
#SBATCH -a 0-3 # array size
#SBATCH --cpus-per-task=8 # Number of CPU cores per task
#SBATCH --gres=gpu:1 # Number of GPU per task
#SBATCH -J DARTS-Genotype-Continue # sets the job name. If not
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CUPS_PER_NODE gpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

source ~/.bashrc
conda activate confopt


genotype_file=$1
continue_folder=$2
continue_runtime=$3
dataset=$4
batch_size=$5
epochs=$6
meta_info=$7
comments=$8

if [ ! -e "$genotype_file" ]; then
  echo "Genotype file does not exist."
  exit 1
fi

if [ ! -d "${continue_folder}/${continue_runtime}" ]; then
    echo "Directory does not exists: ${continue_folder}/${continue_runtime}"
    exit 1
fi

genotype=$(<"$genotype_file")

python scripts/experiments/train_darts_genotype.py \
        --genotype "$genotype" \
        --continue_folder $continue_folder \
        --continue_runtime $continue_runtime \
        --dataset $dataset \
        --seed  $SLURM_ARRAY_TASK_ID \
        --batch-size $batch_size \
        --epochs $epochs \
        --project-name iclr-train-genotypes \
        --meta-info $meta_info \
        --comments $comments \

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
