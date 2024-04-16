Check Memory
==============

Introduction
------------

This script is designed to provide information about processes running on a Unix-like operating system. It allows users to specify the name of a process and provides details such as memory usage and the number of instances of the process currently running. The script requires two input parameters: the path to the script file and the name of the process to monitor.

Usage
-----

To use the script, follow these steps:

1. Open a terminal window.

2. Navigate to the directory where the script is located.

3. Run the script using the following command:

   .. code-block:: bash

      ./script_name.sh <script_path> <process_name>

   Replace ``script_name.sh`` with the actual name of the script file, ``<script_path>`` with the path to the script file you want to monitor, and ``<process_name>`` with the name of the process you want to track.

Examples
--------

Here are some examples demonstrating how to use the script:

1. Monitoring a Python script named ``run_sim.py``:

   .. code-block:: bash

      ./script_name.sh /path/to/run_sim.py python3

   This command will monitor the Python process associated with the ``run_sim.py`` file.

2. Tracking a Python process named ``run_rl_sim.py``:

   .. code-block:: bash

      ./script_name.sh /path/to/run_rl_sim.py python3

   This command will monitor the Python process associated with the ``run_rl_sim.py`` file.

Output
------

The script will provide the following output:

- Total memory used by the specified process, in megabytes (MB) and gigabytes (GB).
- Total number of instances of the specified process currently running.

For example:

.. code-block:: bash

   Total memory used is: 100 MB or 0.0976562 GB
   Total number of processes used by my_script.py is: 1

This output indicates that the specified process is currently using 100 megabytes of memory and there is one instance of the process running.

Additional Notes
----------------

- Make sure the script file exists at the specified path.
- Ensure that you have appropriate permissions to access and execute the script file.

``script_name.sh``: The name of the script file.

``<script_path>``: The path to the script file to monitor.

``<process_name>``: The name of the process to track.

``python3``: The Python interpreter associated with the process.

``/path/to/run_sim.py``: Example path to a Python script file.

``/path/to/run_rl_sim.py``: Example path to another Python script file.

