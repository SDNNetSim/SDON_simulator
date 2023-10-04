#!/bin/bash

#SBATCH -p cpu-long
#SBATCH -c 28
#SBATCH --mem=16000
#SBATCH -t 3-00:00:00
#SBATCH -o slurm-%j.out

# shellcheck disable=SC2164
cd
# shellcheck disable=SC2164
cd /work/pi_vinod_vokkarane_uml_edu/git/SDN_Simulator/

beta_array=("0.000001" "0.1" "0.2" "0.4" "0.6" "0.8")
k_array=("1" "2" "3" "4" "5")

python run_sim.py --route_method xt_aware --num_requests 10000 --beta ${beta_array[$SLURM_ARRAY_TASK_ID]}
