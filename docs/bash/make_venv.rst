Make Venv
==========

Introduction
------------

This Bash script automates the creation of a Python virtual environment in a specified directory using a
specified Python version. It ensures that the target directory exists, the specified Python version is available,
and then creates a virtual environment named 'venv' in the target directory using the provided Python version.

Usage
-----

To use the script, follow these steps:

1. Open a terminal window.

2. Navigate to the directory where the script is located.

3. Run the script using the following command:

   .. code-block:: bash

      ./make_venv.sh <target_directory> <python_version>

   Replace ``make_venv.sh`` with the actual name of the script file, ``<target_directory>`` with the path to the directory where you want to create the virtual environment, and ``<python_version>`` with the desired Python version to use for the virtual environment.

   **Example**:

   .. code-block:: bash

      ./make_venv.sh ~/my_project python3.8

   This command will create a virtual environment named 'venv' in the directory ``~/my_project`` using Python version 3.8.

Output
------

The script provides the following output:

- A message indicating the successful creation of the virtual environment.

For example:

.. code-block:: bash

   Virtual environment 'venv' created in '~/my_project' using python3.8!

Additional Notes
----------------

- Ensure that you have appropriate permissions to create directories and execute the script.
- Make sure the specified Python version is installed on your system or is accessible via the system's PATH.

``make_venv.sh``: The name of the script file.

``<target_directory>``: The path to the directory where the virtual environment will be created.

``<python_version>``: The Python version to use for the virtual environment.

``~/my_project``: Example path to the target directory for creating the virtual environment.
