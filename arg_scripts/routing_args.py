# pylint: disable=too-few-public-methods

class RoutingProps:
    """
    Main properties used for the routing.py script.
    """

    def __init__(self):
        # TODO: paths_list changed to paths_matrix
        self.paths_matrix = []  # Matrix of potential paths for a single request
        # TODO: mod_formats_list to mod_formats_matrix
        self.mod_formats_matrix = []  # Modulation formats corresponding to each path in paths_matrix
        self.weights_list = []  # Keeping track of one weight of the path (Length, XT, Etc.)

        self.input_power = 1e-3  # Power in Watts
        self.freq_spacing = 12.5e9  # Frequency spacing in Hz
        self.mci_worst = 6.3349755556585961e-027  # Worst-case mutual coupling interference value
        self.max_link_length = None  # Maximum link length in km
        self.span_len = 100.0  # Length of a span in km
        self.max_span = None  # Maximum number of spans in the network

    def __repr__(self):
        return f"EmptyProps({self.__dict__})"
