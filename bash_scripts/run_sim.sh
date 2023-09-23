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

cat <<-PYTHONSCRIPT >mypythonscript.py
    #!/usr/bin/env python3
    import sys
    job_names = ["job1", "job2"]
    job_index = int(sys.args[1])
    print(job_names[job_index])
PYTHONSCRIPT

python run_sim.py $SLURM_ARRAY_TASK_ID
