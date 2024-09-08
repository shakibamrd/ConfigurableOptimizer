#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH -o logs/%x.%N.%j.out # STDOUT
#SBATCH -e logs/%x.%N.%j.err # STDERR
#SBATCH -a 1-4 # array size
#SBATCH -J ConfoptExperiment # sets the job name. If not
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CUPS_PER_NODE gpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

source ~/.bashrc
conda activate confopt

searchspace=$1
sampler=$2
epochs=$3
k=$4
warm_epochs=$5

python scripts/experiments/run_experiment.py \
        --searchspace $searchspace \
        --dataset "cifar10" \
        --sampler $sampler \
        --epochs $epochs \
        --seed $SLURM_ARRAY_TASK_ID \
        --partial-connection \
        --partial-connection-k $k \
        --partial-connection-warm-epochs $warm_epochs \
        --arch-attention-enabled \
        --project-name iclr-experiments \
        --rq arch-attention-and-partial-connections \
        --comment "for non-darts searchspaces, the trainer config is wrong"

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
