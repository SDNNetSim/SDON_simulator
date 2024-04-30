#!/bin/bash

#SBATCH -p cpu-preempt
#SBATCH -c 1
#SBATCH -G 0
#SBATCH --mem=50000
#SBATCH -t 2-00:00:00
#SBATCH -o slurm-%j.out

# shellcheck disable=SC2164
# cd
# shellcheck disable=SC2164
# cd /work/pi_vinod_vokkarane_uml_edu/git/sdn_simulator/

# Make and activate virtual environment
#rm -rf venvs/unity_venv/venv
#module load python/3.11.0
#./bash_scripts/make_venv.sh venvs/unity_venv python3.11
# source venvs/unity_venv/venv/bin/activate

# Download requirements
#pip install -r requirements.txt

# Modify StableBaselines3 to register custom environments
#./bash_scripts/register_rl_env.sh ppo SimEnv

# Run AI simulation
# python -m rl_zoo3.train --algo ppo --env SimEnv --conf-file ./ai_scripts/yml/ppo.yml -optimize --n-trials 5 --n-timesteps 20000
#python -m rl_zoo3.train --algo ppo --env SimEnv --conf-file ./ai_scripts/yml/ppo.yml --n-timesteps 20000 --save-freq 10000

# Learning rate, discount factor, epsilon

# Variables for outer loop
var1_list=("0.1" "0.1" "0.2" "0.2" "0.8" "0.8" "0.9" "0.9")
var2_list=("0.99" "0.1" "0.99" "0.1" "0.99" "0.1" "0.99" "0.1")
var3_list=("0.10" "0.10" "0.10" "0.10" "0.10" "0.10" "0.10" "0.10")

python run_rl_sim.py --learn_rate "${var1_list[${SLURM_ARRAY_TASK_ID}]}" --discount_factor "${var2_list[${SLURM_ARRAY_TASK_ID}]}" --epsilon_start "${var3_list[${SLURM_ARRAY_TASK_ID}]}"

# Run regular simulation
# python run_sim.py --max_segments 1 --k_paths 3
