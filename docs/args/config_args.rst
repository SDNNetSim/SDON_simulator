Configuration Arguments
=======================

.. automodule:: arg_scripts.config_args
    :members:
    :undoc-members:

.. list-table::
   :widths: 25 25 50
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
     -