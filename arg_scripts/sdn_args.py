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
        self.modulation_list = []  # List of modulation formats used by a single request
        self.core_list = []  # List of cores used (typically for light-segment slicing)
        self.xt_list = []  # List of crosstalk calculations for a single request
        self.num_trans = None  # Number of transponders a single request has used
        self.arrive = None  # Arrival time for a single request
        self.depart = None  # Departure time for a single request
        self.request_type = None  # Determines arrival or departure
        self.slots_needed = None  # Slots needed for the current request
        self.single_core = False  # Whether to force single-core
        self.block_reason = None  # Reason for blocking a request
        self.mod_formats_dict = None  # List of valid modulation formats for this bandwidth

        self.stat_key_list = ['modulation_list', 'xt_list', 'core_list']  # Statistical keys used to save results

    def update_params(self, stat_key: str, spectrum_key: str, spectrum_obj: object, stat_value: int = None):
        """
        Update lists to track statistics of routed requests or general network metrics.

        :param stat_key: Statistical key to update.
        :param spectrum_key: Spectrum key to update.
        :param spectrum_obj: Spectrum assignment main object.
        :param stat_value: A statistical value related to the stat key, it may vary.
        """
        if stat_key == 'modulation_list':
            self.modulation_list.append(spectrum_obj.spectrum_props[spectrum_key])
        elif stat_key == 'xt_list':
            self.xt_list.append(spectrum_obj.spectrum_props[spectrum_key])
        elif stat_key == 'core_list':
            self.core_list.append(spectrum_obj.spectrum_props[spectrum_key])
        elif stat_key == 'req_id':
            self.req_id = stat_value
        elif stat_key == 'source':
            self.source = stat_value
        elif stat_key == 'destination':
            self.destination = stat_value
        elif stat_key == 'arrive':
            self.arrive = stat_value
        elif stat_key == 'depart':
            self.depart = stat_value
        elif stat_key == 'request_type':
            self.request_type = stat_value
        elif stat_key == 'bandwidth':
            self.bandwidth = stat_value
        elif stat_key == 'mod_formats':
            self.mod_formats_dict = stat_value
        else:
            raise NotImplementedError(f'Got an unaccounted stat key: {stat_key}')

    def reset_params(self):
        """
        Reset select lists used to track statistics.
        """
        self.modulation_list = list()
        self.xt_list = list()
        self.core_list = list()

    def get_data(self, stat_key: str):
        """
        Retrieves desired data in properties.

        :return: The desired data.
        """
        if stat_key == 'modulation_list':
            resp = self.modulation_list
        elif stat_key == 'xt_list':
            resp = self.xt_list
        elif stat_key == 'core_list':
            resp = self.core_list
        else:
            raise NotImplementedError

        return resp

    def __repr__(self):
        return f"EmptyProps({self.__dict__})"
