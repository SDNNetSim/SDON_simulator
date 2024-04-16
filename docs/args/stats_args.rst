Statistics Arguments
=======================

This documentation page focuses on the various statistics arguments employed in EON simulations.
These arguments track crucial network parameters, such as blocking probability, providing valuable insights
into network behavior. By understanding these arguments, you can gain a deeper understanding of your EON's
performance limitations and optimize its configuration for optimal results.

.. automodule:: arg_scripts.stats_args
    :members:
    :undoc-members:

.. list-table:: empty_props
   :widths: 25 25
   :header-rows: 1

   * - Argument Name
     - Description
   * - snapshots_dict
     - Relevant statistics at certain snapshots in the simulation
   * - cores_dict
     - Tracks how often each core was allocated on
   * - weights_dict
     - Tracks weights of each allocated request, a weight may be length, cross-talk, hops, etc.
   * - mods_used_dict
     - Tracks modulation formats used
   * - block_bw_dict
     - Tracks how often each bandwidth was blocked
   * - block_reasons_dict
     - Tracks the reasons for blocking any request
   * - sim_block_list
     - Blocking probabilities for each iteration
   * - trans_list
     - Total number of transponders used
   * - hops_list
     - Number of hops each request made
   * - lengths_list
     - Length of each path taken in the simulation
   * - route_times_list
     - Tracks time taken to route each request
   * - xt_list
     - Tracks intra or inter core cross-talk for allocated requests
