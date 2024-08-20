Run Sim
========

Introduction
------------

This Bash script sets up and submits one or more batch jobs using the SLURM workload manager.
It configures job parameters such as the partition, number of CPU cores, memory requirements, runtime, output file,
and environment setup. The script is particularly useful for running AI simulations or other computationally
intensive tasks in a distributed manner on a high-performance computing (HPC) cluster.

Usage
-----

To use the script, follow these steps:

1. Open a text editor.

2. Copy and paste the script contents into the text editor.

3. Customize the script as needed, especially the commands to set up the environment and run simulations.

4. Save the file with a meaningful name, such as ``run_sim.sh``.

5. Open a terminal window.

6. Navigate to the directory where the script is located.

7. Submit the job using the SLURM command:

   .. code-block:: bash

      sbatch run_sim.sh

Output
------

The script sets up and submits one or more batch jobs to the SLURM workload manager.
The output of the jobs will depend on the commands and parameters specified within the script, including any
output file paths provided via the ``#SBATCH -o`` directive.

Additional Notes
----------------

- Make sure you have appropriate permissions to execute the script and submit jobs to the SLURM scheduler.
- Customize the SLURM directives (``#SBATCH`` lines) according to your cluster's configuration and job requirements.
- Adjust the environment setup and simulation commands to match your specific workflow and software dependencies.

``run_sim.sh``: The name of the script file.

``sbatch``: The SLURM command used to submit batch jobs.

``#SBATCH``: SLURM directives for configuring job parameters.

``--algo ppo``: Example algorithm parameter used in the simulation command.

``--env SimEnv``: Example environment parameter used in the simulation command.

``python -m rl_zoo3.train``: Example command for running AI simulations with rl_zoo3.

``python run_sim.py``: Example command for running regular simulations.

