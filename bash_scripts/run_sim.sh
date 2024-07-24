#!/bin/bash

#SBATCH -p cpu-long
#SBATCH -c 1
#SBATCH -G 0
#SBATCH --mem=16000
#SBATCH -t 1-00:00:00
#SBATCH -o slurm-%j.out

# This script is designed to run a non-artificial intelligence simulation on the Unity cluster at UMass Amherst.
# Users can provide custom parameters via the command line using -- before the parameter list.

# Ensure the script stops if any command fails
set -e

# Change to the home directory and then to the SDN simulator directory
# shellcheck disable=SC2164
cd
# shellcheck disable=SC2164
cd /work/pi_vinod_vokkarane_uml_edu/git/sdn_simulator/

# Load the required Python module
module load python/3.11.0

# Activate the virtual environment
source venvs/unity_venv/venv/bin/activate

# Install the required Python packages
pip install -r requirements.txt

# Register custom reinforcement learning environments in Stable Baselines 3
./bash_scripts/register_rl_env.sh ppo SimEnv

# Define default parameters for the simulation
NETWORK="Pan-European"
TRAIN_FILE_PATH="Pan-European/0531/22_00_16_630834"
ML_MODEL="knn"

# Run the non-artificial intelligence simulation with the specified parameters
python run_sim.py --network "$NETWORK" --train_file_path "$TRAIN_FILE_PATH" --ml_model "$ML_MODEL"
