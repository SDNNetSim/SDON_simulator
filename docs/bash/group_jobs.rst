Group Jobs
============

Introduction
------------

This Bash script retrieves resource usage information for specified users' jobs on a Unix-like operating system using
the SLURM workload manager. It takes a start date as input and calculates resource usage from that date until the
present moment. The script also displays resource usage for currently running jobs. Users can customize the list of
users for which resource usage is calculated.

Usage
-----

To use the script, follow these steps:

1. Open a terminal window.

2. Navigate to the directory where the script is located.

3. Run the script using the following command:

   .. code-block:: bash

      ./group_jobs.sh <start_date>

   Replace ``<start_date>`` with the desired start date for resource usage calculation. The start date should be provided in a format recognized by the ``date`` command. For example, you can use ``'7 days ago'`` as the start date.

   **Example**:

   .. code-block:: bash

      ./group_jobs.sh '7 days ago'

   This command will calculate resource usage starting from 7 days ago until the present moment.

Output
------

The script provides the following output:

- Resource usage for each specified user, including the total number of CPUs used, memory usage in gigabytes (GB) and megabytes (MB), and the number of nodes used.

For example:

.. code-block:: bash

   Resource usage from 2023-04-05 to now:
   User ryan_mccann_student_uml_edu is using 8 CPUs, 13.20 GB (or 13527 MB) of memory, and 2 nodes.
   User arash_rezaee_student_uml_edu is using 4 CPUs, 6.60 GB (or 6771 MB) of memory, and 1 nodes.
   ...

Additional Notes
----------------

- Make sure the script file exists in the specified location.
- Ensure that you have appropriate permissions to execute the script file.
- Customize the ``GROUP_USERS`` array in the script to include the desired users for resource usage calculation.

``group_jobs.sh``: The name of the script file.

``<start_date>``: The start date for resource usage calculation.

``'7 days ago'``: Example of a start date for resource usage calculation.

