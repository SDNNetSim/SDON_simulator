import networkx as nx
import numpy as np


class Routing:
    def __init__(self, source, destination, physical_topology, network_spec_db, slots_needed=None):
        self.path = None

        self.source = source
        self.destination = destination
        self.physical_topology = physical_topology
        self.network_spec_db = network_spec_db
        self.slots_needed = slots_needed

        self.paths_list = list()

    def find_least_cong_route(self):
        # Sort dictionary by number of slots occupied key and return the first one
        sorted_paths_list = sorted(self.paths_list, key=lambda d: d['link_info']['slots_taken'])

        return sorted_paths_list[0]['path']

    def find_most_cong_link(self, path):
        res_dict = {'link': None, 'slots_taken': -1}

        for i in range(len(path) - 1):
            cores_matrix = self.network_spec_db[(path[i]), path[i + 1]]['cores_matrix']
            link_num = self.network_spec_db[(path[i]), path[i + 1]]['link_num']
            slots_taken = 0

            for core_num, core_arr in enumerate(cores_matrix):
                slots_taken += len(np.where(core_arr == 1)[0])
            if slots_taken > res_dict['slots_taken']:
                res_dict['slots_taken'] = slots_taken
                res_dict['link'] = link_num

        # Link info is information about the most congested link found
        self.paths_list.append({'path': path, 'link_info': res_dict})

    def least_congested_path(self):
        paths_obj = nx.all_simple_paths(G=self.physical_topology, source=self.source, target=self.destination)
        # Sort sub-arrays by length
        paths_matrix = np.array([np.array(y) for x, y in sorted([(len(x), x) for x in paths_obj])], dtype=object)
        min_hops = None

        for i, path in enumerate(paths_matrix):
            num_hops = len(path)
            if i == 0:
                min_hops = num_hops
                self.find_most_cong_link(path)
            else:
                if num_hops <= min_hops + 1:
                    self.find_most_cong_link(path)

        # TODO: Remember that this returns a numpy array (can change)
        return self.find_least_cong_route()
