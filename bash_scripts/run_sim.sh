#!/bin/bash

#SBATCH -p cpu-long
#SBATCH -c 20
#SBATCH -G 0
#SBATCH --mem=40000
#SBATCH -t 4-00:00:00
#SBATCH -o slurm-%j.out
#SBATCH --array=0-3

# This script is designed to run a non-artificial intelligence simulation on the Unity cluster at UMass Amherst.
# Users can provide custom parameters via the command line using -- before the parameter list.

# Ensure the script stops if any command fails
set -e

# Change to the home directory and then to the SDN simulator directory
# shellcheck disable=SC2164
cd
# shellcheck disable=SC2164
cd /home/arash_rezaee_student_uml_edu/Git/SDON_simulator/

# Load the required Python module
module load python/3.11.0

# Activate the virtual environment
source venvs/unity_venv/venv/bin/activate

# Install the required Python packages
pip install -r requirements.txt

# define parameters
allocation_methods=("first_fit" "last_fit")  
spectrum_allocation_priorities=("CSB" "BSC")  

total_combinations=$(( ${#allocation_methods[@]} * ${#spectrum_allocation_priorities[@]} ))

# Ensure the task ID is within the range of total combinations
if [ "$SLURM_ARRAY_TASK_ID" -ge "$total_combinations" ]; then
  echo "SLURM_ARRAY_TASK_ID out of range."
  exit 1
fi

# Calculate the allocation method and spectrum allocation priority index based on SLURM_ARRAY_TASK_ID
allocation_method_index=$(( SLURM_ARRAY_TASK_ID / ${#spectrum_allocation_priorities[@]} ))
spectrum_priority_index=$(( SLURM_ARRAY_TASK_ID % ${#spectrum_allocation_priorities[@]} ))

# Get the allocation method and spectrum priority for the current task
allocation_method=${allocation_methods[$allocation_method_index]}
spectrum_priority=${spectrum_allocation_priorities[$spectrum_priority_index]}

# Print the combination being used
echo "Running simulation with allocation_method: $allocation_method and spectrum_allocation_priority: $spectrum_priority"

# Run the simulation
python run_sim.py --allocation_method "$allocation_method" --spectrum_allocation_priority "$spectrum_priority"