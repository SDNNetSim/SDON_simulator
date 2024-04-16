Register RL Env
================

Introduction
------------

This Bash script registers a custom environment with the Gymnasium library for reinforcement learning.
It takes two arguments: the algorithm (``<algo>``) and the environment name (``<env_name>``). The script then invokes
a Python script to perform the registration and appends registration details to a Python file.

Usage
-----

To use the script, follow these steps:

1. Open a terminal window.

2. Navigate to the directory where the script is located.

3. Run the script using the following command:

   .. code-block:: bash

      ./register_rl_env.sh <algo> <env_name>

   Replace ``<algo>`` with the algorithm name and ``<env_name>`` with the name of the custom environment you want to register.

   **Example**:

   .. code-block:: bash

      ./register_rl_env.sh PPO custom_env

   This command will register a custom environment named 'custom_env' for the PPO algorithm.

Output
------

The script performs the following actions:

- Registers the specified custom environment with the Gymnasium library for reinforcement learning.
- Appends registration details to a Python file located in the virtual environment's directory.

Additional Notes
----------------

- Make sure you have appropriate permissions to execute the script.
- Ensure that the required Python environment is set up properly and accessible to the script.

``register_rl_env.sh``: The name of the script file.

``<algo>``: The name of the algorithm for which you want to register the custom environment.

``<env_name>``: The name of the custom environment you want to register.

``PPO``: Example algorithm name.

``custom_env``: Example name of the custom environment.

