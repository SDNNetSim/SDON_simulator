Spectrum Arguments
====================

In elastic optical networks (EONs), spectrum assignment refers to the dynamic allocation of flexible frequency slices
to accommodate varying traffic demands. Effective spectrum assignment is essential for maximizing spectral efficiency
and network performance. This documentation page explores the arguments used in spectrum assignment algorithms for
EONs. The information here will help you understand the factors that influence how spectral resources are distributed
across the network.

.. automodule:: arg_scripts.spectrum_args
    :members:
    :undoc-members:

.. list-table:: empty_props
   :widths: 25 25
   :header-rows: 1

   * - Argument Name
     - Description
   * - path_list
     - A list of nodes to be allocated on
   * - slots_needed
     - The number of frequency slots the request needs
   * - forced_core
     - A flag to force the request on a specific core
   * - is_free
     - Flag to determine if all spectrum along the path are free or not
   * - modulation
     - Modulation format of the request
   * - xt_cost
     - Cross-talk cost of the request, may be intra or inter core.
   * - cores_matrix
     - Contains spectrum for each core along a link
   * - rev_cores_matrix
     - The reversed direction of cores_matrix
   * - core_num
     - Core number the request was allocated in
   * - forced_index
     - To force the request to a certain frequency channel, mainly used for reinforcement learning
   * - start_slot
     - Start slot/channel the request was allocated on
   * - end_slot
     - End slot/channel the request was allocated on