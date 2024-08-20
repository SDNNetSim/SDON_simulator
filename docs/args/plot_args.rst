Plot Arguments
==============

The simulator provides a wide range of configuration arguments to fine-tune the appearance of generated plots.
These arguments control elements such as the displayed data, axis labels, plot titles, and color schemes.

.. automodule:: arg_scripts.plot_args
    :members:
    :undoc-members:

.. list-table:: empty_props
   :widths: 25 25
   :header-rows: 1

   * - Argument Name
     - Description
   * - sim_info_dict
     - Relevant information regarding the simulation
   * - plot_dict
     - Relevant information for plotting
   * - output_dir
     - Directory to save plots
   * - input_dir
     - Directory to read simulation information
   * - sim_num
     - The simulation number e.g., ``s1``
   * - erlang_dict
     - Contains iteration information for each traffic volume
   * - num_requests
     - Number of requests used for the simulation
   * - num_cores
     - Number of cores used for the simulation
   * - color_list
     - Colors a line can use when plotted
   * - style_list
     - Styles a line can use when plotted
   * - marker_list
     - Markers a line can use when plotted
   * - x_tick_list
     - Ticks on the x-axis
   * - title_names
     - Titles for each plot

.. list-table:: empty_plot_dict
   :widths: 25 25
   :header-rows: 1

   * - Argument Name
     - Description
   * - erlang_list
     - List of Erlangs
   * - blocking_list
     - List of blocking probabilities
   * - lengths_list
     - List of average path lengths
   * - hops_list
     - List of average number of hops
   * - occ_slot_matrix
     - List of occupied slots
   * - active_req_matrix
     - List of number of active requests
   * - block_req_matrix
     - List of blocking probabilities at certain request arrivals
   * - req_num_list
     - Request numbers
   * - times_list
     - List of routing times
   * - modulations_dict
     - Modulation formats used
   * - dist_block_list
     - Blocking due to distance
   * - cong_block_list
     - Blocking due to congestion
   * - holding_time
     - Mean holding time for each simulation
   * - cores_per_link
     - Core per link used for each simulation
   * - spectral_slots
     - Slots per core used for each simulation
   * - learn_rate
     - Learning rate used for Q-Learning Algorithm
   * - discount_factor
     - Discount factor used for Q-Learning Algorithm
   * - sum_rewards_list
     - Average rewards Q-Learning algorithm obtained
   * - sum_errors_list
     - Average TD errors Q-Learning algorithm obtained
   * - epsilon_list
     - Information about epsilon decay values
