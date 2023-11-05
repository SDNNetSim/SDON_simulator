#!/bin/bash

#SBATCH -p cpu-long
#SBATCH -c 28
#SBATCH --mem=48000
#SBATCH -t 3-12
#SBATCH -o slurm-%j.out

# shellcheck disable=SC2164
cd
# shellcheck disable=SC2164
cd /home/arash_rezaee_student_uml_edu/Git/SDN_Simulator/


beta_array=("0.000001" "0.1" "0.2" "0.4" "0.6" "0.8")
k_array=("1" "2" "3" "4" "5")

python run_sim.py --sim_type arash --route_method k_shortest_path --k_paths ${k_array[$SLURM_ARRAY_TASK_ID]}
