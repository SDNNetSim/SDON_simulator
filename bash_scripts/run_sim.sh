#!/bin/bash

#SBATCH -p cpu-long
#SBATCH -c 20
#SBATCH --mem=16000
#SBATCH -t 10-00:00:00
#SBATCH -o slurm-%j.out

cd
cd /work/pi_vinod_vokkarane_uml_edu/git/SDN_Simulator/
python run_sim.py
