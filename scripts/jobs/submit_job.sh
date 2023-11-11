#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080 # partition (queue)
#SBATCH --gres=gpu:5
#SBATCH -o logs/%x.%N.%j.out # STDOUT
#SBATCH -e logs/%x.%N.%j.err # STDERR
#SBATCH -a 1-10 # array size
#SBATCH -J Confopt_rs # sets the job name. If not

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CUPS_PER_NODE gpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

source ~/.bashrc
conda activate confopt

python scripts/random_search.py --searchspace "nb201" --num_exp 30 --epochs 5 --wandb_log --seed $SLURM_ARRAY_TASK_ID

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
