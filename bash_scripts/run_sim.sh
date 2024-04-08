#!/bin/bash

#SBATCH -p gpu-preempt
#SBATCH -c 1
#SBATCH -G 1
#SBATCH --mem=80000
#SBATCH -t 2-00:00:00
#SBATCH -o slurm-%j.out

# shellcheck disable=SC2164
#cd
# shellcheck disable=SC2164
#cd /work/pi_vinod_vokkarane_uml_edu/git/sdn_simulator/

# Make and activate virtual environment
#rm -rf venvs/unity_venv/venv
#module load python/3.11.0
#./bash_scripts/make_venv.sh venvs/unity_venv python3.11
source venvs/unity_venv/venv/bin/activate

# Download requirements
#pip install -r requirements.txt

# Modify StableBaselines3 to register custom environments
#./bash_scripts/register_rl_env.sh ppo SimEnv

# Run AI simulation
# python -m rl_zoo3.train --algo ppo --env SimEnv --conf-file ./ai_scripts/yml/ppo.yml -optimize --n-trials 5 --n-timesteps 20000
python -m rl_zoo3.train --algo ppo --env SimEnv --conf-file ./ai_scripts/yml/ppo.yml --n-timesteps 20000 --save-freq 10000
#python run_rl_sim.py

# Run regular simulation
# python run_sim.py --max_segments 1 --k_paths 3
