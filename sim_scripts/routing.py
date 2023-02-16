import networkx as nx
import numpy as np

from useful_functions.sim_functions import get_path_mod, find_path_len


class Routing:
    """
    Contains the routing methods for the simulation.
    """

    def __init__(self, req_id, source, destination, physical_topology, network_spec_db, mod_formats,
                 slots_needed=None, bw=None):  # pylint: disable=invalid-name
        self.path = None

        self.req_id = req_id
        self.source = source
        self.destination = destination
        self.physical_topology = physical_topology
        self.network_spec_db = network_spec_db
        self.slots_needed = slots_needed
        self.bw = bw  # pylint: disable=invalid-name

        self.mod_formats = mod_formats

        self.paths_list = list()

    def find_least_cong_route(self):
        """
        Given a list of dictionaries containing the most congested routes for each path,
        find the least congested route.

        :return: The least congested route
        :rtype: list
        """
        # Sort dictionary by number of free slots, descending (least congested)
        sorted_paths_list = sorted(self.paths_list, key=lambda d: d['link_info']['free_slots'], reverse=True)

        return sorted_paths_list[0]['path']

    def find_most_cong_link(self, path):
        """
        Given a list of nodes, or a path, find the most congested link between all nodes. Count how many
        slots are taken. For multiple cores, the spectrum slots occupied is added for each link.

        :param path: A given path
        :type path: list
        """
        res_dict = {'link': None, 'free_slots': None}

        for i in range(len(path) - 1):
            cores_matrix = self.network_spec_db[(path[i]), path[i + 1]]['cores_matrix']
            link_num = self.network_spec_db[(path[i]), path[i + 1]]['link_num']
            # The total amount of free spectral slots
            free_slots = 0

            for core_num, core_arr in enumerate(cores_matrix):  # pylint: disable=unused-variable
                free_slots += len(np.where(core_arr == 0)[0])
                # We want to find the least amount of free slots
            if res_dict['free_slots'] is None or free_slots < res_dict['free_slots']:
                res_dict['free_slots'] = free_slots
                res_dict['link'] = link_num

        # Link info is information about the most congested link found
        self.paths_list.append({'path': path, 'link_info': res_dict})

    def least_congested_path(self):
        """
        Given a graph with a desired source and destination, implement the least congested pathway algorithm. (Based on
        Arash Rezaee's research paper's assumptions)

        :return: The least congested path
        :rtype: list
        """
        paths_obj = nx.shortest_simple_paths(G=self.physical_topology, source=self.source, target=self.destination)
        min_hops = None

        for i, path in enumerate(paths_obj):
            num_hops = len(path)
            if i == 0:
                min_hops = num_hops
                self.find_most_cong_link(path)
            else:
                if num_hops <= min_hops + 1:
                    self.find_most_cong_link(path)
                else:
                    path = self.find_least_cong_route()
                    return path

        return False, False, False

    def shortest_path(self):
        """
        Given a graph with a desired source and destination, find the shortest path with respect to link lengths.

        :return: The shortest path
        :rtype: list
        """
        paths_obj = nx.shortest_simple_paths(G=self.physical_topology, source=self.source, target=self.destination,
                                             weight='length')

        # Modulation format calculations based on Yue Wang's dissertation
        for path in paths_obj:
            path_len = find_path_len(path, self.physical_topology)
            mod_format = get_path_mod(self.mod_formats, path_len)
            return path, mod_format
