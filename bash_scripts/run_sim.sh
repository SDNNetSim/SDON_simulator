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

# Run simulation
python run_sim.py

# Run AI simulation
