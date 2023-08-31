#!/bin/bash
#SBATCH -p relea_gpu-rtx2080 # mlhiwidlc_gpu-rtx2080     # partition (queue)
#SBATCH -t 0-07:00           # time (D-HH:MM)
#SBATCH -o logs/slurm_logs/%A.%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/slurm_logs/%A.%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J darts-profile              # sets the job name. 
#SBATCH --mem=10G  

# Print some information about the job to STDOUT

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# python -u runner.py --config-file $1

echo submitted ${config_file_seed}
python scripts/darts_runner.py 


echo "DONE";
echo "Finished at $(date)";