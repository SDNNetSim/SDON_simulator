#!/bin/bash

#SBATCH -p cpu-long
#SBATCH -c 20
#SBATCH --mem=36000
#SBATCH -t 10-12
#SBATCH -o slurm-%j.out

# shellcheck disable=SC2164
cd
# shellcheck disable=SC2164
cd /home/arash_rezaee_student_uml_edu/Git/SDN_Simulator/

beta_array=("0.000001" "0.1" "0.2" "0.4" "0.6" "0.8")
k_array=("1" "2" "3" "4" "5")

python run_sim.py --sim_type arash --route_method xt_aware --beta ${beta_array[$SLURM_ARRAY_TASK_ID]}
