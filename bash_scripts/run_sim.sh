#!/bin/bash

#SBATCH -p cpu-long
#SBATCH -c 20
#SBATCH -G 0
#SBATCH --mem=40000
#SBATCH -t 4-00:00:00
#SBATCH -o slurm-%j.out

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

# Define default parameters for the simulation
x="BSC"

# Run the non-artificial intelligence simulation with the specified parameters
python run_sim.py --spectrum_allocation_priority "$x"
