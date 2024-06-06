#!/bin/bash

#SBATCH -p cpu-long
#SBATCH -c 1
#SBATCH -G 0
#SBATCH --mem=16000
#SBATCH -t 0-02:00:00
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

# Variables for outer loop
# Declare arrays for parameter values
learn_rate_list=("0.1" "0.01" "0.5")
disc_factor_list=("0.1" "0.01" "0.9")
reward_list=("1" "10" "100")
penalty_list=("0" "-1" "-10" "-100")

# Calculate total number of combinations
total_combinations=$((${#learn_rate_list[@]} * ${#disc_factor_list[@]} * ${#reward_list[@]} * ${#penalty_list[@]}))

# Iterate through every combination
index=0  # Track the current combination index
for lr in "${learn_rate_list[@]}"; do
    for df in "${disc_factor_list[@]}"; do
        for r in "${reward_list[@]}"; do
            for p in "${penalty_list[@]}"; do

                # Ensure the index matches the SLURM array task ID
                while [ $index -ne $SLURM_ARRAY_TASK_ID ]; do
                    index=$((index + 1)) 
                    if [ $index -eq $total_combinations ]; then
                        index=0  # Reset if we reach the end
                    fi
                done

                # Run the Python script with the current combination
                python run_rl_sim.py --learn_rate $lr --discount_factor $df --reward $r --penalty $p 
                
                # Increment the index for the next combination
                index=$((index + 1))
            done
        done
    done
done

# Run regular simulation
# python run_sim.py --network Pan-European --train_file_path "Pan-European/0531/22_00_16_630834" --ml_model knn
