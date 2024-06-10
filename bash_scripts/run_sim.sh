#!/bin/bash

#SBATCH -p cpu-long
#SBATCH -c 1
#SBATCH -G 0
#SBATCH --mem=16000
#SBATCH -t 1-00:00:00
#SBATCH -o slurm-%j.out

# shellcheck disable=SC2164
cd
# shellcheck disable=SC2164
cd /work/pi_vinod_vokkarane_uml_edu/git/sdn_simulator/

# Make and activate virtual environment
# rm -rf venvs/unity_venv/venv
module load python/3.11.0
# ./bash_scripts/make_venv.sh venvs/unity_venv python3.11
source venvs/unity_venv/venv/bin/activate

# Download requirements
pip install -r requirements.txt

# Modify StableBaselines3 to register custom environments
./bash_scripts/register_rl_env.sh ppo SimEnv

# Learning rate, discount factor, epsilon

# Declare arrays for parameter values 
learn_rate_list=("0.1" "0.1" "0.1" "0.01" "0.01" "0.01" "0.3" "0.3" "0.3")
disc_factor_list=("0.1" "0.01" "0.9" "0.1" "0.01" "0.9" "0.1" "0.01" "0.9")
reward_list=("10" "10" "10" "10" "10" "10" "10" "10" "10")
penalty_list=("-100" "-100" "-100" "-100" "-100" "-100" "-100" "-100" "-100")

# Get parameter values based on SLURM_ARRAY_TASK_ID
lr="${learn_rate_list[$SLURM_ARRAY_TASK_ID]}"
df="${disc_factor_list[$SLURM_ARRAY_TASK_ID]}"
r="${reward_list[$SLURM_ARRAY_TASK_ID]}"
p="${penalty_list[$SLURM_ARRAY_TASK_ID]}"


# Run the Python script with the extracted parameters
python run_rl_sim.py \
       --learn_rate "$lr" \
       --discount_factor "$df" \
       --reward "$r" \
       --penalty "$p" --path_algorithm first_fit --core_algorithm q_learning\
       || echo "Error running Python script for job $SLURM_ARRAY_TASK_ID"

# Run regular simulation
# python run_sim.py --network Pan-European --train_file_path "Pan-European/0531/22_00_16_630834" --ml_model knn
