#!/bin/bash

#SBATCH -p cpu-preempt
#SBATCH -c 1
#SBATCH --mem=20000
#SBATCH -t 1-00:00:00
#SBATCH -o slurm-%j.out

# shellcheck disable=SC2164
cd
# shellcheck disable=SC2164
cd /work/pi_vinod_vokkarane_uml_edu/git/sdn_simulator/

# Make and activate virtual environment
./bash_scripts/make_venv.sh venvs/unity_venv python3.11
source venvs/unity_venv/venv/bin/activate

# Download requirements
pip install -r requirements.txt

# Modify StableBaselines3 to register custom environments
./bash_scripts/register_rl_env.sh custom_dqn DQNSimEnv

# Run AI simulation
python -m rl_zoo3.train --algo dqn --env DQNSimEnv --conf-file ./ai_scripts/yml/custom_dqn.yml --env-kwargs arguments:1,128,10

# Run regular simulation
# python run_sim.py
