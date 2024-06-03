#!/bin/bash

#SBATCH -p cpu-long
#SBATCH -c 13
#SBATCH -G 0
#SBATCH --mem=16000
#SBATCH -t 2-00:00:00
#SBATCH -o slurm-%j.out

# shellcheck disable=SC2164
cd
# shellcheck disable=SC2164
cd /work/pi_vinod_vokkarane_uml_edu/git/sdn_simulator/

# Make and activate virtual environment
# rm -rf venvs/unity_venv/venv
# module load python/3.11.0
# ./bash_scripts/make_venv.sh venvs/unity_venv python3.11
# source venvs/unity_venv/venv/bin/activate

# Download requirements
pip install -r requirements.txt

# Modify StableBaselines3 to register custom environments
# ./bash_scripts/register_rl_env.sh ppo SimEnv

# Learning rate, discount factor, epsilon

# Variables for outer loop
learn_rate_list=("0.01" "0.01" "0.9" "0.9" "0.01" "0.01" "0.9" "0.9" "0.01" "0.01" "0.9" "0.9")
disc_factor_list=("0.9" "0.1" "0.9" "0.1" "0.9" "0.1" "0.9" "0.1" "0.9" "0.1" "0.9" "0.1")
epsilon_start_list=("0.30" "0.30" "0.30" "0.30" "0.30" "0.30" "0.30" "0.30" "0.30" "0.30" "0.30" "0.30")
reward_list=("1.0" "1.0" "1.0" "1.0" "1.0" "1.0" "1.0" "1.0" "1.0" "1.0" "1.0" "1.0")
penalty_list=("-1.0" "-1.0" "-1.0" "-1.0" "0.0" "0.0" "0.0" "0.0" "-10.0" "-10.0" "-10.0" "-10.0")

INDEX=$SLURM_ARRAY_TASK_ID
# Extract values based on index
LEARN_RATE=${learn_rate_list[$INDEX]}
DISC_FACTOR=${disc_factor_list[$INDEX]}
EPSILON_START=${epsilon_start_list[$INDEX]}
REWARD=${reward_list[$INDEX]}
PENALTY=${penalty_list[$INDEX]}

# python run_rl_sim.py --path_algorithm q_learning --core_algorithm first_fit --spectrum_algorithm first_fit --learn_rate $LEARN_RATE --discount_factor $DISC_FACTOR --epsilon_start $EPSILON_START --reward $REWARD --penalty $PENALTY

# Run regular simulation
python run_sim.py --network Pan-European
