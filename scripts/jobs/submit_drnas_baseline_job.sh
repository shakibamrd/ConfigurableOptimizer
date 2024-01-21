#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080 #relea_gpu-rtx2080 #mlhiwidlc_gpu-rtx2080 # partition (queue)
#SBATCH -o logs/%x.%N.%j.out # STDOUT
#SBATCH -e logs/%x.%N.%j.err # STDERR
#SBATCH -a 1-4 # array size
#SBATCH -J drnas_base # sets the job name. If not
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CUPS_PER_NODE gpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

source ~/.bashrc
conda activate confopt

python examples/experiment_drnas.py --search_epochs 100 --eval_epochs 300 --seed $SLURM_ARRAY_TASK_ID --searchspace "nb201" --dataset "cifar10"

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
