#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH -o logs/%x.%N.%j.out # STDOUT
#SBATCH -e logs/%x.%N.%j.err # STDERR
#SBATCH -a 0-3 # array size
#SBATCH --cpus-per-task=8 # Number of CPU cores per task
#SBATCH -J DARTS-Genotype-100-epochs # sets the job name. If not
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CUPS_PER_NODE gpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

source ~/.bashrc
conda activate confopt

dataset=cifar10

genotype_folder=$1
sampler=darts
dataset=cifar10
meta_info="'DARTS-Genotypes-100-Epochs'"
comments="'None'"

# Extract 4 genotypes from the folder
if [ ! -d "$genotype_folder" ]; then
  echo "Directory does not exist."
  exit 1
fi

genotypes=($(find "$genotype_folder" -type f -name "*.txt"))
genotype_count=${#genotypes[@]}

if [ "$genotype_count" -ne 4 ]; then
  echo "Error: Expected 4 genotype files, but found $genotype_count."
  exit 1
fi

genotype_1_file=${genotypes[0]}
genotype_2_file=${genotypes[1]}
genotype_3_file=${genotypes[2]}
genotype_4_file=${genotypes[3]}

genotype_1=$(<"$genotype_1_file")
genotype_2=$(<"$genotype_2_file")
genotype_3=$(<"$genotype_3_file")
genotype_4=$(<"$genotype_4_file")

experiment_group=${genotype_folder##*/}

python scripts/experiments/get_darts_best_genotype.py \
        --experiment-group $experiment_group \
        --genotype-1 "$genotype_1" \
        --genotype-2 "$genotype_2" \
        --genotype-3 "$genotype_3" \
        --genotype-4 "$genotype_4" \
        --dataset $dataset \
        --seed  $SLURM_ARRAY_TASK_ID\
        --project-name iclr-experiments \
        --meta-info $meta_info \
        --comments $comments \

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
