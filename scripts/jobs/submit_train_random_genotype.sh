#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH -o logs/%x.%N.%j.out # STDOUT
#SBATCH -e logs/%x.%N.%j.err # STDERR
#SBATCH -a 0-30 # array size
#SBATCH --cpus-per-task=8 # Number of CPU cores per task
#SBATCH --gres=gpu:1 # Number of GPU per task
#SBATCH -J Small-DARTS-Genotype # sets the job name. If not
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CUPS_PER_NODE gpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

source ~/.bashrc
conda activate confopt

python scripts/experiments/train_random_darts_genotype.py --seed $SLURM_ARRAY_TASK_ID  --channels 36 --layers 4 --train-portion 0.4
# python scripts/experiments/train_random_darts_genotype.py --seed $SLURM_ARRAY_TASK_ID  --channels 36 --layers 4 --train-portion 0.1

# python scripts/experiments/train_random_darts_genotype.py --seed $SLURM_ARRAY_TASK_ID  --channels 16 --layers 4 --train-portion 0.4
# python scripts/experiments/train_random_darts_genotype.py --seed $SLURM_ARRAY_TASK_ID  --channels 16 --layers 4 --train-portion 0.1

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
