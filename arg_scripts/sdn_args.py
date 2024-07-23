# pylint: disable=too-few-public-methods

class SDNProps:
    """
    Main properties used for the sdn_controller.py script.
    """

    def __init__(self):
        self.path_list = None  # List of nodes for the current request
        self.was_routed = None  # Flag to determine successful route
        self.topology = None  # Networkx topology
        self.net_spec_dict = None  # Current network spectrum database
        self.req_id = None  # Current request ID number
        self.source = None  # Source node
        self.destination = None  # Destination node
        self.bandwidth = None  # Current bandwidth
        self.bandwidth_list = []  # Multiple bandwidths used (typically for light-segment slicing)
        self.modulation_list = []  # List of valid modulation formats
        self.core_list = []  # List of cores used (typically for light-segment slicing)
        self.xt_list = []  # List of crosstalk calculations for a single request
        self.stat_key_list = ['modulation_list', 'xt_list', 'core_list']  # Statistical keys used to save results
        self.num_trans = None  # Number of transponders a single request has used
        self.slots_needed = None  # Slots needed for the current request
        self.single_core = False  # Whether to force single-core
        self.block_reason = None  # Reason for blocking a request

    def __repr__(self):
        return f"EmptyProps({self.__dict__})"
