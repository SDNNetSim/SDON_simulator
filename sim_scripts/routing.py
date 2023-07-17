# Standard library imports
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

    def q_routing(self):
        """
        Given a graph with a source and destination, find a path using the Q-learning RL algorithm.

        :return: A tuple containing the "best" path according to the Q-table and the modulation format
        :rtype: tuple
        """
        pass
