Routing Arguments
==================

Effective network routing is crucial for ensuring data reaches its intended destination efficiently. This reference
guide provides a comprehensive overview of the various arguments used to configure routing protocols and network
traffic. The table below details each argument, its purpose, and its impact on how your network routes data packets.

.. automodule:: arg_scripts.routing_args
    :members:
    :undoc-members:

.. list-table:: empty_props
   :widths: 25 25
   :header-rows: 1

   * - Argument Name
     - Description
   * - paths_list
     - Potential paths to route a request (returned from routing)
   * - mod_formats_list
     - Modulation formats associated with each potential path
   * - weights_list
     - Weights associated with each potential path, could be length, cross-talk cost, hops, etc.
   * - input_power
     - ``Arash``
   * - freq_spacing
     - ``Arash``
   * - mci_worst
     - ``Arash``
   * - max_link_length
     - Maximum link length in the topology
   * - span_len
     - Length of a single span
   * - max_span
     - Maximum number of spans for a path in the current topology
