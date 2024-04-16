Kill Script
============

Introduction
------------

This Bash script terminates a running script identified by its file name. It takes the path to the script file as
input and terminates any processes associated with that script. This script is particularly useful when you need to
stop a long-running process or script.

Usage
-----

To use the script, follow these steps:

1. Open a terminal window.

2. Navigate to the directory where the script is located.

3. Run the script using the following command:

   .. code-block:: bash

      ./kill_script.sh <script_path>

   Replace ``<script_path>`` with the path to the script file you want to terminate.

   **Example**:

   .. code-block:: bash

      ./kill_script.sh /path/to/your_script.py

   This command will terminate the script identified by ``kill_script.sh``.

Output
------

The script provides the following output:

- A message indicating that the specified script has been terminated.

For example:

.. code-block:: bash

   your_script.sh has been killed.

Additional Notes
----------------

- Make sure the script file exists at the specified path.
- Ensure that you have appropriate permissions to execute the script file.

``kill_script.sh``: The name of the script file.

``<script_path>``: The path to the script file to terminate.

``/path/to/your_script.py``: Example path to a script file that you want to terminate.

