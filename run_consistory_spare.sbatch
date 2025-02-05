#!/bin/bash
#SBATCH --job-name=white_woman # Job name
#SBATCH --output=%x_%j_%N.out # Standard output file: <name>_<job id>_<nodename>.out 
# #SBATCH --error=%x_%j_%N.err # Standard error file: <name>_<job id>_<nodename>.err
#SBATCH --partition=comm # Partition to submit the job to
#SBATCH --nodelist=crunky # Specific nodes to run the job on
#SBATCH --gres=gpu:a6000:1 # Generic resources (e.g. GPUs) per node
#SBATCH --cpus-per-task=8 # Number of CPUs per task
#SBATCH --mem=128G # Minimum memory per task

# Optional
# SBATCH --ntasks-per-node=1 # Number of tasks per node
# #SBATCH --time=01:00:00 # Time limit for the job
# #SBATCH --nodes=2 # Number of nodes requested
# #SBATCH --exclude=node3,node4 # Nodes to exclude from the job
# #SBATCH --requeue # Allow job to be requeued

ENV_ACTIVATE_PATH=/opt/anaconda3/bin/activate
source /home/${USER}/.bashrc

# Write the name of the virtual env to use
ENV_NAME=consistory

# Job execution commands
source $ENV_ACTIVATE_PATH $ENV_NAME # conda activate

python consistory_CLI.py --condition 0 --config config_1_asian.yaml --concept_override "asian woman" --hair_override "long hair"

#python consistory_CLI.py --condition 0 --config config_2_black.yaml --concept_override "black woman" --hair_override "long hair"

#python consistory_CLI.py --condition 0 --config config_3_white.yaml --concept_override "white woman" --hair_override "long hair"