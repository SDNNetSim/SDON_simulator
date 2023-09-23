#!/bin/bash

#SBATCH -p cpu-long
#SBATCH -c 16
#SBATCH --mem=16000
#SBATCH -t 4-00:00:00
#SBATCH -o slurm-%j.out

# shellcheck disable=SC2164
cd
# shellcheck disable=SC2164
cd /work/pi_vinod_vokkarane_uml_edu/git/SDN_Simulator/

beta_array=("0.000001" "0.1")

python run_sim.py --beta beta_array[$SLURM_ARRAY_TASK_ID]
