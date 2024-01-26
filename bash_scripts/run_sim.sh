#!/bin/bash

#SBATCH -p cpu-preempt
#SBATCH -c 1
#SBATCH --mem=20000
#SBATCH -t 1-00:00:00
#SBATCH -o slurm-%j.out

# shellcheck disable=SC2164
cd
# shellcheck disable=SC2164
cd /work/pi_vinod_vokkarane_uml_edu/git/SDN_Simulator/

python run_sim.py
