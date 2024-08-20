Priority
=========

Introduction
------------

This Bash script retrieves priority information for pending jobs of a specified user in a particular SLURM partition.
It identifies the priority of the user's pending jobs, the total number of pending jobs with higher priority,
and the highest priority job in the partition's queue.

Usage
-----

To use the script, follow these steps:

1. Open a terminal window.

2. Navigate to the directory where the script is located.

3. Run the script using the following command:

   .. code-block:: bash

      ./script_name.sh <username> <partition>

   Replace ``<username>`` with the username of the user whose job priorities you want to check, and ``<partition>`` with the name of the SLURM partition.

   **Example**:

   .. code-block:: bash

      ./priority.sh ryan_mccann_student_uml_edu cpu-long

   This command will retrieve priority information for pending jobs of user 'ryan_mccann_student_uml_edu' in the 'cpu-long' partition.

Output
------

The script provides the following output:

- The priority of the specified user's pending jobs in the specified partition.
- The highest priority job in the queue of the specified partition.
- The total number of pending jobs with higher priority than the user's highest priority job.

For example:

.. code-block:: bash

   User: ryan_mccann_student_uml_edu has priority 100 in the 'cpu-long' partition, and the highest priority in the queue is currently 120.
   5 jobs have a higher priority than ryan_mccann_student_uml_edu's highest priority job in the 'cpu-long' partition.

Additional Notes
----------------

- Ensure that SLURM is installed and properly configured on your system.
- Make sure you have appropriate permissions to access SLURM resources and execute the script.

``priority.sh``: The name of the script file.

``<username>``: The username of the user whose job priorities you want to check.

``<partition>``: The name of the SLURM partition.

``ryan_mccann_student_uml_edu``: Example username of the user whose job priorities you want to check.

``cpu-long``: Example name of the SLURM partition.

