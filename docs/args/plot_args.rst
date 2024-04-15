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
   * - (25, QPSK)
     - 1
