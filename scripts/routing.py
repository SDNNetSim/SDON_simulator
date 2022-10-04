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

        self.paths_dict = dict()

    def find_least_cong_route(self):
        # Sort dictionary by number of slots occupied key and return the first one
        res = sorted(self.paths_dict.items(), key=lambda x: x[1]['slots_taken'])
        print(res)
        return list()

    def find_most_cong_link(self, path):
        # Would just be comparing the lengths of finding the non-zero slots of numpy arrays
        res_dict = {'path': path, 'link': None, 'core': None, 'slots_taken': -1}

        for i in range(len(path) - 1):
            cores_matrix = self.network_spec_db[(path[i]), path[i+1]]['cores_matrix']
            link_num = self.network_spec_db[(path[i]), path[i+1]]['link_num']

            for core_num, core_arr in enumerate(cores_matrix):
                slots_taken = len(np.where(core_arr == 1)[0])
                if slots_taken > res_dict['slots_taken']:
                    res_dict['slots_taken'] = slots_taken
                    res_dict['core'] = core_num
                    res_dict['link'] = link_num

        self.paths_dict[path] = res_dict

    def least_congested_path(self):
        paths_obj = nx.all_simple_paths(G=self.physical_topology, source=self.source, target=self.destination)
        # Sort sub-arrays by length
        paths_matrix = np.array([np.array(y) for x, y in sorted([(len(x), x) for x in paths_obj])], dtype=object)
        min_hops = None

        for i, path in enumerate(paths_matrix):
            if i == 0:
                min_hops = len(path)
                self.find_most_cong_link(path)
            else:
                num_hops = len(path)
                if num_hops <= min_hops + 1:
                    self.find_most_cong_link(path)
                else:
                    return self.find_least_cong_route()
