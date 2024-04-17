#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080 #relea_gpu-rtx2080 #mlhiwidlc_gpu-rtx2080 # partition (queue)
#SBATCH -o logs/%x.%N.%j.out # STDOUT
#SBATCH -e logs/%x.%N.%j.err # STDERR
#SBATCH -a 1-3 # array size
#SBATCH -J robust_darts_lora # sets the job name. TODO: Change the name of the experiment
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CUPS_PER_NODE gpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

source ~/.bashrc
conda activate confopt

# use lora experiment name 
#darts sampler
python scripts/lora/run_robust_darts.py --use_lora --lora_warm_epoch 10 --lora_rank 4 --lora_merge_weights --sampler "darts" --wandb_log --space "s1"
# python scripts/lora/run_robust_darts.py --use_lora --lora_warm_epoch 10 --lora_rank 4 --lora_merge_weights --sampler "darts" --wandb_log --space "s2"
# python scripts/lora/run_robust_darts.py --use_lora --lora_warm_epoch 10 --lora_rank 4 --lora_merge_weights --sampler "darts" --wandb_log --space "s3"
# python scripts/lora/run_robust_darts.py --use_lora --lora_warm_epoch 10 --lora_rank 4 --lora_merge_weights --sampler "darts" --wandb_log --space "s4"

# drnas sampler
# python scripts/lora/run_robust_darts.py --sampler "drnas" --wandb_log --space "s1"
# python scripts/lora/run_robust_darts.py --sampler "drnas" --wandb_log --space "s2"
# python scripts/lora/run_robust_darts.py --sampler "drnas" --wandb_log --space "s3"
# python scripts/lora/run_robust_darts.py --sampler "drnas" --wandb_log --space "s4"
# python scripts/lora/run_robust_darts.py --use_lora --lora_warm_epoch 10 --lora_rank 4 --lora_merge_weights --sampler "drnas" --wandb_log --space "s1"
# python scripts/lora/run_robust_darts.py --use_lora --lora_warm_epoch 10 --lora_rank 4 --lora_merge_weights --sampler "drnas" --wandb_log --space "s2"
# python scripts/lora/run_robust_darts.py --use_lora --lora_warm_epoch 10 --lora_rank 4 --lora_merge_weights --sampler "drnas" --wandb_log --space "s3"
# python scripts/lora/run_robust_darts.py --use_lora --lora_warm_epoch 10 --lora_rank 4 --lora_merge_weights --sampler "drnas" --wandb_log --space "s4"

# gdas sampler
# python scripts/lora/run_robust_darts.py --sampler "gdas" --wandb_log --space "s1"
# python scripts/lora/run_robust_darts.py --sampler "gdas" --wandb_log --space "s2"
# python scripts/lora/run_robust_darts.py --sampler "gdas" --wandb_log --space "s3"
# python scripts/lora/run_robust_darts.py --sampler "gdas" --wandb_log --space "s4"
# python scripts/lora/run_robust_darts.py --use_lora --lora_warm_epoch 10 --lora_rank 4 --lora_merge_weights --sampler "gdas" --wandb_log --space "s1"
# python scripts/lora/run_robust_darts.py --use_lora --lora_warm_epoch 10 --lora_rank 4 --lora_merge_weights --sampler "gdas" --wandb_log --space "s2"
# python scripts/lora/run_robust_darts.py --use_lora --lora_warm_epoch 10 --lora_rank 4 --lora_merge_weights --sampler "gdas" --wandb_log --space "s3"
# python scripts/lora/run_robust_darts.py --use_lora --lora_warm_epoch 10 --lora_rank 4 --lora_merge_weights --sampler "gdas" --wandb_log --space "s4"

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
