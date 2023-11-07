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



python run_sim.py --sim_type arash --route_method xt_aware --xt_type without_length
