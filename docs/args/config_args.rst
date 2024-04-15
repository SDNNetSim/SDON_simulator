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
     - The simulation assumptions for calculating the Erlang
     - ``arash`` | ``yue``
   * - holding_time
     - The mean holding time for request generation
     - Any floating point value
   * - arrival_rate
     - The inter-arrival time for request generation
     - Any floating point value
   * - thread_erlangs
     - Whether to run the traffic volumnes specified in parallel or not
     - ``True`` | ``False``
   * - guard_slots
     - The amount of frequency channels dedicated to the guard band
     - Any integer value
   * - holding_time
     - The mean holding time
     - N/A
