#!/bin/bash
#SBATCH --job-name=asian_man # Job name
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

# Loop through all condition directories
for condition_dir in config/condition_*; do
    condition_num=$(echo $condition_dir | grep -o '[0-9]*$')
    
    # Loop through all yaml files in each condition directory
    for config_file in $condition_dir/config_[0-9]_*.yaml; do
        if [ -f "$config_file" ]; then
            config_name=$(basename "$config_file")
            echo "Running: $config_file"
            python consistory_CLI.py --condition $condition_num --config $config_name --concept_override "asian man"
        fi
    done
done