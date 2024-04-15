Reinforcement Learning Arguments
=================================

The reinforcement learning (RL) component of the simulator offers a set of configuration arguments for fine-tuning
the agent's behavior and training process. These arguments govern aspects like the action space, observation
space, and tracking important statistics.


.. automodule:: arg_scripts.rl_args
    :members:
    :undoc-members:

.. list-table:: empty_ai_props
   :widths: 25 25
   :header-rows: 1

   * - Argument Name
     - Description
   * - k_paths
     - Number of paths the RL agent has to choose from
   * - cores_per_link
     - Number of cores the RL agent has to choose from
   * - spectral_slots
     - Number of slots the DRL agent has to choose from
   * - num_nodes
     - Number of nodes of the topology
   * - bandwidth_list
     - List of bandwidths used in the simulation
   * - arrival_list
     - Contains every arrival request
   * - depart_list
     - Contains every departure request
   * - mock_sdn_dict
     - Used in place of the sdn_dict for certain functionality
   * - source
     - Source node
   * - destination
     - Destination node
   * - paths_list
     - List of potential paths
   * - path_index
     - Index of the path chosen
   * - chosen_path
     - The chosen path
   * - core_index
     - Index of the core chosen

.. list-table:: empty_q_props
   :widths: 25 25
   :header-rows: 1

   * - Argument Name
     - Description
   * - epsilon
     - Current epsilon the agent is using
   * - epsilon_start
     - Number of paths the DRL agent has to choose from

.. list-table:: empty_drl_props
   :widths: 25 25
   :header-rows: 1

   * - Argument Name
     - Description
   * - (25, QPSK)
     - 1
