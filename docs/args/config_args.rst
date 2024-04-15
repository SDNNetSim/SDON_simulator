Configuration Arguments
=======================

.. automodule:: arg_scripts.config_args
    :members:
    :undoc-members:

.. list-table::
   :widths: 50 50 50
   :header-rows: 1

   * - Argument Name
     - Description
     - Valid Input
   * - sim_type
     - Simulation assumptions for calculating the Erlang
     - ``arash`` | ``yue``
   * - holding_time
     - Mean holding time for request generation
     - Any floating point value
   * - arrival_rate
     - Inter-arrival time for request generation
     - Any floating point value
   * - thread_erlangs
     - Run the traffic volumes in parallel or not
     - ``True`` | ``False``
   * - guard_slots
     - Frequency channels dedicated to the guard band
     - Any integer value
   * - num_requests
     - Requests to generate for a single iteration
     - Any integer value
   * - request_distribution
     - Bandwidth distribution of requests
     - Any floating point values that add up to 1.0
   * - max_iters
     - Maximum iterations to run
     - Any integer value
   * - max_segments
     - Maximum segments for a single request
     - Any integer value
   * - dynamic_lps
     - Use dynamic light path/segment slicing or not
     - ``True`` | ``False``
   * - allocation_method
     - Method for assigning a request to a spectrum
     - ``best_fit`` | ``first_fit`` | ``last_fit`` | ``priority_first`` | ``priority_last`` | ``xt_aware``
   * - route_method
     - Method for routing a request
     - ``nli_aware`` | ``xt_aware`` | ``least_congested`` | ``shortest_path`` | ``k_shortest_path``
   * - save_snapshots
     - To save information at certain request intervals
     - ``True`` | ``False``
   * - snapshot_step
     - Interval for saving snapshot results
     - Any integer value that is a multiple of the number of requests
   * - print_step
     - Interval for printing simulator information
     - Any integer value that is a multiple of the number of requests
   * - network
     - Network topology
     - ``USNet`` | ``NSFNet`` | ``Pan-European``
   * - spectral_slots
     - Spectral slots per core on a given link
     - Any integer value
   * - bw_per_slot
     - The bandwidth (GHz) for each frequency slot
     - Any floating point value
   * - cores_per_link
     - Number of cores for every link in the topology
     - Any integer value
   * - const_link_weight
     - Sets all link weights to 1
     - ``True`` | ``False``
   * - file_type
     - File structure to save to
     - ``json``
   * - erlangs
     - Used from ``arash`` type simulations to determine erlang distribution
     - Any range of integer values
   * - requested_xt
     - ``Arash``
     - ``Arash``
   * - xt_noise
     - ``Arash``
     - ``Arash``
   * - theta
     - ``Arash``
     - ``Arash``
   * - egn_model
     - ``Arash``
     - ``Arash``
   * - phi
     - ``Arash``
     - ``Arash``
   * - snr_type
     - ``Arash``
     - ``Arash``
   * - xt_type
     - ``Arash``
     - ``Arash``
   * - beta
     - ``Arash``
     - ``Arash``
   * - input power
     - ``Arash``
     - ``Arash``
   * - ai_algorithm
     - Use QL or a specified DRL algorithm
     - ``q_learning`` | ``ppo`` | ``a2c`` | ``dqn``
   * - learn_rate
     - Learning rate for q-learning algorithm
     - Any floating point value
   * - discount_factor
     - Discount factor for q-learning algorithm
     - Any floating point value
   * - epsilon_start
     - Where epsilon starts for q-learning algorithm
     - Any floating point value
   * - epsilon_end
     - Where epsilon will end for q-learning algorithm
     - Any floating point value

StableBaselines3 and RL Baselines3 Zoo
---------------------------------------

Parameters within these libraries should work when running the simulator via command line. For more information on which
parameters exist and their descriptions, please see: `StableBaselines3 <https://stable-baselines3.readthedocs.io/en/master/>`_ and
`RL Baselines3 Zoo <https://www.example.com/>`_ docs.
