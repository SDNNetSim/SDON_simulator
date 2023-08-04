# Standard library imports
import math
from typing import List

# Third-party library imports
import networkx as nx
import numpy as np

# Local application imports
from useful_functions.sim_functions import get_path_mod, find_path_len


class Routing:
    """
    This class contains methods for routing packets in a network topology.
    """

    def __init__(self, source: int = None, destination: int = None, topology: nx.Graph = None, net_spec_db: dict = None,
                 mod_formats: dict = None, slots_needed: int = None, bandwidth: float = None):
        """
        Initializes the Routing class.

        :param source: The source node ID.
        :type source: int

        :param destination: The destination node ID.
        :type destination: int

        :param topology: The network topology represented as a NetworkX Graph object.
        :type topology: nx.Graph

        :param net_spec_db: A database of network spectrum.
        :type net_spec_db: dict

        :param mod_formats: A dict of available modulation formats and their potential reach.
        :type mod_formats: dict

        :param slots_needed: The number of slots needed for the connection.
        :type slots_needed: int

        :param bandwidth: The required bandwidth for the connection.
        :type bandwidth: float
        """
        self.source = source
        self.destination = destination
        self.topology = topology
        self.net_spec_db = net_spec_db
        self.slots_needed = slots_needed
        self.bandwidth = bandwidth
        self.mod_formats = mod_formats

        # The nodes of a pathway for a given request
        self.path = []
        # A list of potential paths
        self.paths_list = []
        # Constants related to non-linear impairment calculations
        self.input_power = 1e-3
        self.freq_spacing = 12.5e9
        self.span_len = 100
        self.mci = 0
        self.mci_w = 6.3349755556585961e-027
        # TODO: Update for efficiency
        self.max_link = max(nx.get_edge_attributes(topology, 'length').values(), default=0.0)

    def find_least_cong_path(self):
        """
        Finds the least congested path from the list of available paths.

        :return: The least congested path
        :rtype: List[int]
        """
        # Sort dictionary by number of free slots, descending (least congested)
        sorted_paths_list = sorted(self.paths_list, key=lambda d: d['link_info']['free_slots'], reverse=True)

        return sorted_paths_list[0]['path']

    def find_most_cong_link(self, path: List[int]):
        """
        Given a path, find the most congested link between all nodes. Count how many slots are taken.
        For multiple cores, the number of spectrum slots occupied is added for each link.

        :param path: The path to analyze
        :type path: list[int]

        :return: The congestion level of the most congested link in the path
        :rtype: int
        """
        # Initialize variables to keep track of the most congested link found
        most_congested_link = None
        most_congested_slots = -1

        # Iterate over all links in the path
        for i in range(len(path) - 1):
            link = self.net_spec_db[(path[i], path[i + 1])]
            cores_matrix = link['cores_matrix']

            # Iterate over all cores in the link
            for core_arr in cores_matrix:
                free_slots = np.sum(core_arr == 0)

                # Check if the current core is more congested than the most congested core found so far
                if free_slots < most_congested_slots or most_congested_link is None:
                    most_congested_slots = free_slots
                    most_congested_link = link

        # Update the list of potential paths with information about the most congested link in the path
        self.paths_list.append(
            {'path': path, 'link_info': {'link': most_congested_link, 'free_slots': most_congested_slots}})

        return most_congested_slots

    def least_congested_path(self):
        """
        Implementation of the least congested pathway algorithm, based on Arash Rezaee's research paper's assumptions.

        :return: The least congested path, or a tuple indicating that the algorithm failed to find a valid path
        :rtype: list or tuple[bool, bool, bool]
        """
        # Use NetworkX to find all simple paths between the source and destination nodes
        all_paths = nx.shortest_simple_paths(self.topology, self.source, self.destination)

        min_hops = None

        # Iterate over all simple paths found until the number of hops exceeds the minimum hops plus one
        for i, path in all_paths:
            num_hops = len(path)
            if i == 0:
                min_hops = num_hops
                self.find_most_cong_link(path)
            else:
                if num_hops <= min_hops + 1:
                    self.find_most_cong_link(path)
                # We exceeded minimum hops plus one, return the best path
                else:
                    least_cong_path = self.find_least_cong_path()
                    return least_cong_path

        # If no valid path was found, return a tuple indicating failure
        return False, False, False

    def shortest_path(self, weight='length'):
        """
        Given a graph with a desired source and destination, find the shortest path with respect to link lengths.

        :param weight: What to consider as a weight for the shortest path, either NLI cost or link length.
        :type weight: str

        :return: A tuple containing the shortest path and its modulation format
        :rtype: tuple
        """
        # This networkx function will always return the shortest paths in order
        paths_obj = nx.shortest_simple_paths(G=self.topology, source=self.source, target=self.destination,
                                             weight=weight)

        for path in paths_obj:
            path_len = find_path_len(path, self.topology)
            mod_format = get_path_mod(self.mod_formats, path_len)

            return path, mod_format

    def _find_channel_mci(self, num_spans, center_freq, taken_channels):
        mci = 0
        for channel in taken_channels:
            # The current center frequency for the occupied channel
            # TODO: DEBUG: in the following lines instead of self.slots_needed  we need number of slots of the taken channel!!!!!
            curr_freq = (channel[0] * self.freq_spacing) + ((self.slots_needed * self.freq_spacing) / 2)
            bandwidth = self.slots_needed * self.freq_spacing
            # Power spectral density
            power_spec_dens = self.input_power / bandwidth

            mci += (power_spec_dens ** 2) * math.log(abs((abs(center_freq - curr_freq) + (bandwidth / 2)) / (
                    abs(center_freq - curr_freq) - (bandwidth / 2))))

        return (mci / self.mci_w) * num_spans

    def _find_link_cost(self, num_spans, free_channels, taken_channels):
        # Non-linear impairment cost calculation
        nli_cost = 0

        # Update MCI for available channel
        for channel in free_channels:
            # Calculate the center frequency for the open channel
            center_freq = (channel[0] * self.freq_spacing) + ((self.slots_needed * self.freq_spacing) / 2)
            nli_cost += self._find_channel_mci(num_spans=num_spans, taken_channels=taken_channels,
                                               center_freq=center_freq)

        if len(free_channels) == 0:
            return 1000

        link_cost = nli_cost / len(free_channels)

        return link_cost

    def _find_channels(self, link, check_free):
        if check_free:
            indexes = np.where(self.net_spec_db[link]['cores_matrix'][0] == 0)[0]
        else:
            indexes = np.where(self.net_spec_db[link]['cores_matrix'][0] != 0)[0]

        channels = []
        curr_channel = []
        for i, idx in enumerate(indexes):
            if i == 0:
                curr_channel.append(idx)
            elif idx == indexes[i - 1] + 1:
                curr_channel.append(idx)
                if len(curr_channel) == self.slots_needed:
                    channels.append(curr_channel)
                    curr_channel = []
            else:
                curr_channel = []

        # Check if the last group forms a subarray
        if len(curr_channel) == 3:
            channels.append(curr_channel)

        return channels

    # TODO: Only support for single core
    # TODO: Maximum number of allowed nodes
    def nli_aware(self, slots_needed=None, beta=None):
        self.slots_needed = slots_needed
        span_len = 100.0

        for link in self.net_spec_db:
            source, destination = link[0], link[1]
            num_spans = self.topology[source][destination]['length'] / span_len

            free_channels = self._find_channels(link=link, check_free=True)
            taken_channels = self._find_channels(link=link, check_free=False)

            nli_cost = self._find_link_cost(num_spans=num_spans, free_channels=free_channels,
                                            taken_channels=taken_channels)
            # Tradeoff between link length and the non-linear impairment cost
            link_cost = (beta * (self.topology[source][destination]['length'] / self.max_link)) + \
                        ( (1 - beta) * nli_cost)

            self.topology[source][destination]['nli_cost'] = link_cost
            self.topology[destination][source]['nli_cost'] = link_cost  # TODO: please check it and if it doesn't need this code remove it

        # TODO: How do you assign a modulation format if lengths are non-linear impairments?
        #   - Assuming a static modulation format, not sure about bit-rate generations
        return self.shortest_path(weight='nli_cost')
