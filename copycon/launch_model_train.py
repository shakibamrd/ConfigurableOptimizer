# mypy: ignore-errors

import argparse
import logging

from slurmpilot import (
    SlurmWrapper,
    JobCreationInfo,
    default_cluster_and_partition,
    unify,
)


def read_config(file_path: str) -> dict[str, str]:
    config = {}
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#"):  # Ignore empty lines and comments
                key, value = line.split("=", 1)
                config[key.strip()] = value.strip().strip('"')  # Remove surrounding quotes if any
    return config


if __name__ == "__main__":
    # Parse command-line arguments for experiment name and number of seeds.
    parser = argparse.ArgumentParser(
        description="Launch experiment jobs with multiple seeds."
    )
    parser.add_argument("--optimizer", required=True, type=str)
    parser.add_argument("--subspace", required=True, type=str)
    parser.add_argument("--opset", required=True, type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--hpsets", type=str, required=True, help="hyperparameter sets to use to train this model")
    parser.add_argument("--seed", required=True, type=str,)
    parser.add_argument("--epochs", type=int, required=True, help="number of epochs to train")
    parser.add_argument("--searchspace", type=str, default="darts")
    parser.add_argument("--other", type=str, required=True, help="other optimizer (FairDARTS, PC-DARTS etc)")
    parser.add_argument("--tag", type=str, required=True, help="tag for the experiment")
    parser.add_argument("--genotypes_folder", type=str, default="genotypes")
    args = parser.parse_args()

    args_dict = vars(args)

    experiment_name = f"train_model-{args.optimizer}-{args.subspace}-{args.opset}"

    python_args = []
    for hpset in args.hpsets.split(","):
        args = " ".join([f"--{k} {v}" for k, v in args_dict.items() if k != "hpsets"])
        args += f" --hpset {hpset}"
        python_args.append(args)

    logging.basicConfig(level=logging.INFO)
    cluster, partition = default_cluster_and_partition()

    # Create a unique jobname using the provided experiment name.
    jobname = unify(f"darts-bench-suite-exps/{experiment_name}", method="coolname")

    config = read_config('config.cfg')
    conda_env = config.get('conda_env_name', 'confopt')

    slurm = SlurmWrapper(clusters=[cluster])
    jobinfo = JobCreationInfo(
        cluster=cluster,
        partition=partition,
        jobname=jobname,
        entrypoint="run_model_train.py",
        # Generate python_args based on the number of seeds provided.
        python_args=python_args,
        python_binary="python",
        n_cpus=8,
        max_runtime_minutes=60 * 8,
        # Pass environment variables to the running script.
        bash_setup_command=f"source ~/.bash_profile; conda activate {conda_env}",
        # n_concurrent_jobs=3,
        env = config,
    )
    jobid = slurm.schedule_job(jobinfo)

    slurm.wait_completion(jobname=jobname, max_seconds=3)
    print(slurm.job_creation_metadata(jobname))
    print(slurm.status([jobname]))

    print("Logs of the latest job:")
    slurm.print_log(jobname=jobname)
