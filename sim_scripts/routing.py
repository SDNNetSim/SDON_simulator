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

    def __init__(self, source: int, destination: int, topology: nx.Graph, net_spec_db: dict, mod_formats: dict,
                 slots_needed: int = None, bandwidth: float = None):
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

    def find_least_cong_route(self):
        """
        Finds the least congested route from the list of available paths.

        :return: The least congested route
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

        # Initialize variables to keep track of the best path and its congestion level
        best_path = None
        least_congested = np.inf

        # Iterate over all simple paths found
        for path in all_paths:
            # Check if the current path is better than the best path found so far
            congestion = self.find_most_cong_link(path)
            if congestion < least_congested:
                least_congested = congestion
                best_path = path

            # Check if the congestion of the best path found so far is below a threshold
            if least_congested <= self.slots_needed:
                return best_path

        # If no valid path was found, return a tuple indicating failure
        return False, False, False

    def shortest_path(self):
        """
        Given a graph with a desired source and destination, find the shortest path with respect to link lengths.
        Modulation format calculations based on Yue Wang's dissertation.

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
