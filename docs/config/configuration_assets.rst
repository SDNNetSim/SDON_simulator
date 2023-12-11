Configuration Assets
=======================

Configurable settings within (.ini) files - Lack of inclusion of the asset in the utilized config (.ini) will result in the asset defaulting.

**Network Configuration**

1. network: A string representing the type or configuration of the network under consideration.

**Warnings Configuration**

1. warnings: A boolean value indicating whether warnings should be enabled or disabled, based on the string-to-boolean conversion.

**Resource Holding Time Configuration**

1. holding_time: A floating-point number representing the duration for which a resource or connection should be reserved or held.

**Erlangs Configuration**

1. erlangs: The specified erlangs to be tested.

**Thread Erlangs Configuration**

1. thread_erlangs: A boolean value indicating whether thread Erlangs are to be considered, based on string-to-boolean conversion.

**Request Processing Configuration**

1. num_requests: An integer specifying the number of requests or transactions to be processed.

**Iteration Configuration**

1. max_iters: An integer defining the maximum number of iterations or cycles in a process.

**Spectral Slots Configuration**

1. spectral_slots: An integer representing the number of spectral slots available in the network.

**Bandwidth Allocation Configuration**

1. bw_per_slot: A floating-point number indicating the bandwidth allocated per spectral slot.

**Processing Cores Configuration**

1. cores_per_link: An integer specifying the number of processing cores allocated per network link.

**Constant Link Weight Configuration**

1. const_link_weight: A boolean value determining whether a constant link weight is applied, based on string-to-boolean conversion.

**Guard Slots Configuration**

1. guard_slots: An integer representing the number of reserved slots as a safety margin.

**Maximum Segments Configuration**

1. max_segments: An integer defining the maximum number of segments in a network.

**Dynamic Logical Paths Configuration**

1.dynamic_lps: A boolean value indicating whether dynamic logical paths are used, based on string-to-boolean conversion.

**Resource Allocation Method Configuration**

1. allocation_method: A string specifying the method used for resource allocation.

**Network Routing Method Configuration**

1. route_method: A string indicating the method employed for determining network routes.

**Request Distribution Configuration**

1. request_distribution: A function to evaluate the distribution of requests.

**Beta Parameter Configuration**

1. beta: A floating-point number representing a parameter often used in mathematical models.

**Snapshots Configuration**

1. save_snapshots: A boolean value indicating whether snapshots should be saved, based on string-to-boolean conversion.

**Component Type Configuration**

1. xt_type: A string denoting the type or category of a particular component or feature in the network.
