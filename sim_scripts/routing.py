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
        # Channel frequency spacing
        self.freq_spacing = 12.5e9
        # Multi-channel interference worst case scenario
        self.mci_w = 6.3349755556585961e-027
        # Maximum link length for the network topology
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
        for i, path in enumerate(all_paths):
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

    def shortest_path(self):
        """
        Given a graph with a desired source and destination, find the shortest path with respect to link lengths.

        :return: A tuple containing the shortest path and its modulation format
        :rtype: tuple
        """
        # This networkx function will always return the shortest paths in order
        paths_obj = nx.shortest_simple_paths(G=self.topology, source=self.source, target=self.destination,
                                             weight='length')

        for path in paths_obj:
            path_len = find_path_len(path, self.topology)
            mod_format = get_path_mod(self.mod_formats, path_len)

            return path, mod_format

    def _least_nli_path(self):
        """
        Selects the path with the least amount of NLI cost.

        :return: The path and modulation format chosen
        :rtype: tuple
        """
        # This networkx function will always return the shortest paths in order
        paths_obj = nx.shortest_simple_paths(G=self.topology, source=self.source, target=self.destination,
                                             weight='nli_cost')

        for path in paths_obj:
            # TODO: Change always a constant
            mod_format = 'QPSK'

            return path, mod_format

    def _find_channel_mci(self, num_spans: float, center_freq: float, taken_channels: list):
        """
        For a given super-channel calculate the multi-channel interference.

        :param num_spans: The number of spans for the link.
        :type num_spans: float

        :param center_freq: The calculated center frequency of the channel.
        :type center_freq: float

        :param taken_channels: A matrix of indexes of the occupied super-channels.
        :type taken_channels: list

        :return: The calculated MCI for the channel.
        :rtype: float
        """
        mci = 0
        for channel in taken_channels:
            # The current center frequency for the occupied channel
            # TODO: Change self.slots_needed to be based on the number of slots the taken channel is
            curr_freq = (channel[0] * self.freq_spacing) + ((self.slots_needed * self.freq_spacing) / 2)
            bandwidth = self.slots_needed * self.freq_spacing
            # Power spectral density
            power_spec_dens = self.input_power / bandwidth

            mci += (power_spec_dens ** 2) * math.log(abs((abs(center_freq - curr_freq) + (bandwidth / 2)) / (
                    abs(center_freq - curr_freq) - (bandwidth / 2))))

        mci = (mci / self.mci_w) * num_spans
        return mci

    def _find_link_cost(self, num_spans: float, free_channels: list, taken_channels: list):
        """
        Find the NLI cost for a link.

        :param num_spans: The number of spans for the given link.
        :type num_spans: float

        :param free_channels: The indexes for all free super-channels on the link.
        :type free_channels: list

        :param taken_channels: The indexes for all occupied super-channels on the link.
        :type taken_channels: list

        :return: The final NLI score calculated.
        :rtype float
        """
        # Non-linear impairment cost calculation
        nli_cost = 0

        # Update MCI for available channel
        for channel in free_channels:
            # Calculate the center frequency for the open channel
            center_freq = (channel[0] * self.freq_spacing) + ((self.slots_needed * self.freq_spacing) / 2)
            nli_cost += self._find_channel_mci(num_spans=num_spans, taken_channels=taken_channels,
                                               center_freq=center_freq)

        # A constant score of 1000 if the link is fully congested
        if len(free_channels) == 0:
            return 1000.0

        link_cost = nli_cost / len(free_channels)

        return link_cost

    def _find_channels(self, link: int, check_free: bool):
        """
        Finds the number of free or occupied channels on any given link.

        :param link: The link number to search for channels on.
        :type link: int

        :param check_free: A flag to check for unoccupied or occupied channels.
        :type check_free: bool

        :return: A matrix containing the indexes to occupied or unoccupied super channels on the link.
        :rtype: list
        """
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
        if len(curr_channel) == self.slots_needed:
            channels.append(curr_channel)

        return channels

    def nli_aware(self, slots_needed: int, beta: float):
        """
        Assigns a non-linear impairment score to every link in the network and selects a path with the least amount of
        NLI cost.

        :param slots_needed: The number of slots needed for the request
        :type slots_needed: int

        :param beta: A tunable parameter used to consider how much the NLI cost vs. link length will be considered.
        :type beta: float

        :return: The path from source to destination with the least amount of NLI cost.
        """
        self.slots_needed = slots_needed
        # Length for one span in km
        span_len = 100.0

        for link in self.net_spec_db:
            source, destination = link[0], link[1]
            num_spans = self.topology[source][destination]['length'] / span_len

            free_channels = self._find_channels(link=link, check_free=True)
            taken_channels = self._find_channels(link=link, check_free=False)

            nli_cost = self._find_link_cost(num_spans=num_spans, free_channels=free_channels,
                                            taken_channels=taken_channels)
            # Tradeoff between link length and the non-linear impairment cost
            link_cost = ( beta * (self.topology[source][destination]['length'] / self.max_link)) + \
                        ((1 - beta) * nli_cost)

            self.topology[source][destination]['nli_cost'] = link_cost

        return self._least_nli_path()
