import networkx as nx
import numpy as np


class Routing:
    """
    Contains the routing methods for the simulation.
    """

    def __init__(self, source, destination, physical_topology, network_spec_db, mod_formats,
                 slots_needed=None):
        self.path = None

        self.source = source
        self.destination = destination
        self.physical_topology = physical_topology
        self.network_spec_db = network_spec_db
        self.slots_needed = slots_needed

        self.mod_formats = mod_formats

        self.paths_list = list()

    def find_least_cong_route(self):
        """
        Given a list of dictionaries containing the most congested routes for each path,
        find the least congested route.

        :return: The least congested route
        :rtype: list
        """
        # Sort dictionary by number of slots occupied key and return the first one
        sorted_paths_list = sorted(self.paths_list, key=lambda d: d['link_info']['slots_taken'])

        return sorted_paths_list[0]['path']

    def find_most_cong_link(self, path):
        """
        Given a list of nodes, or a path, find the most congested link between all nodes. Count how many
        slots are taken. For multiple cores, the spectrum slots occupied is added for each link.

        :param path: A given path
        :type path: list
        """
        res_dict = {'link': None, 'slots_taken': -1}

        for i in range(len(path) - 1):
            cores_matrix = self.network_spec_db[(path[i]), path[i + 1]]['cores_matrix']
            link_num = self.network_spec_db[(path[i]), path[i + 1]]['link_num']
            slots_taken = 0

            for core_num, core_arr in enumerate(cores_matrix):  # pylint: disable=unused-variable
                slots_taken += len(np.where(core_arr == 1)[0])
            if slots_taken > res_dict['slots_taken']:
                res_dict['slots_taken'] = slots_taken
                res_dict['link'] = link_num

        # Link info is information about the most congested link found
        self.paths_list.append({'path': path, 'link_info': res_dict})

    def least_congested_path(self):
        """
        Given a graph with a desired source and destination, find the least congested pathway.

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
                    return self.find_least_cong_route()

        return False

    # TODO: Start with modulation method with shortest length and see if it can fit
    # TODO: Choose modulation format based on length (ranked by highest modulation format)
    def shortest_path(self):
        """
        Given a graph with a desired source and destination, find the shortest path with respect to link lengths.

        :return: The shortest path
        :rtype: list
        """
        paths_obj = nx.shortest_simple_paths(G=self.physical_topology, source=self.source, target=self.destination,
                                             weight='length')
        path_len = 0
        for path in paths_obj:
            for i in range(0, len(path) - 1):
                path_len += self.physical_topology[path[i]][path[i + 1]]['length']

            # TODO: Can they be equal?
            # It's important to check modulation formats in this order
            if self.mod_formats['64-QAM']['max_length'] > path_len:
                mod_format = '64-QAM'
            elif self.mod_formats['16-QAM']['max_length'] > path_len:
                mod_format = '16-QAM'
            elif self.mod_formats['QPSK']['max_length'] > path_len:
                mod_format = 'QPSK'
            else:
                # TODO: Should we look for other paths or block? Block for now.
                return False
            return path, mod_format
