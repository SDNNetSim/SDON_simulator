import networkx as nx
import numpy as np

from arg_scripts.routing_args import empty_props
from helper_scripts.sim_helpers import find_path_len, get_path_mod
from helper_scripts.xt_helpers import get_nli_cost


class Routing:
    """
    This class contains methods related to routing network requests.
    """

    def __init__(self, engine_props: dict, sdn_props: dict):
        self.engine_props = engine_props
        self.sdn_props = sdn_props
        self.route_props = empty_props

    def _find_most_cong_link(self, node_list: list):
        most_cong_link = None
        most_cong_slots = -1

        for i in range(len(node_list) - 1):
            link = self.sdn_props['net_spec_db'][(node_list[i], node_list[i + 1])]
            cores_matrix = link['cores_matrix']

            for core_arr in cores_matrix:
                free_slots = np.sum(core_arr == 0)
                if free_slots < most_cong_slots or most_cong_link is None:
                    most_cong_slots = free_slots
                    most_cong_link = link

        return most_cong_slots

    def _find_least_cong_path(self):
        # Sort dictionary by number of free slots, descending
        sorted_paths_list = sorted(self.route_props['paths_list'], key=lambda d: d['link_info']['free_slots'],
                                   reverse=True)

        return sorted_paths_list[0]['path']

    def find_least_cong_path(self):
        """
        Find the least congested path in the network.

        :return: The least congested path.
        :rtype: list
        """
        all_paths_obj = nx.shortest_simple_paths(self.sdn_props['topology'], self.route_props['source'],
                                                 self.route_props['destination'])
        min_hops = None
        least_cong_path = False

        for i, path in enumerate(all_paths_obj):
            num_hops = len(path)
            if i == 0:
                min_hops = num_hops
                self._find_most_cong_link(path)
            else:
                if num_hops <= min_hops + 1:
                    self._find_most_cong_link(path)
                # We exceeded minimum hops plus one, return the best path
                else:
                    least_cong_path = self._find_least_cong_path()

        self.route_props['paths_list'].append(least_cong_path)
        # TODO: Constant QPSK format
        self.route_props['mod_formats_list'].append('QPSK')
        # TODO: Not sure what to put here
        self.route_props['weights_list'].append(None)

    def find_least_weight_path(self, weight: str):
        """
        Find the path with respect to a given weight.

        :param weight: Determines the weight to consider for finding the path.
        """
        paths_obj = nx.shortest_simple_paths(G=self.sdn_props['topology'], source=self.route_props['source'],
                                             target=self.route_props['destination'], weight=weight)

        for path in paths_obj:
            # If cross-talk, our path weight is the summation across the path
            if weight == 'xt_cost':
                resp_weight = sum(self.sdn_props['topology'][path[i]][path[i + 1]][weight]
                                  for i in range(len(path) - 1))
            else:
                resp_weight = find_path_len(path=path, topology=self.sdn_props['topology'])

            self.route_props['weights_list'].append(resp_weight)
            mod_format = get_path_mod(self.sdn_props['mod_formats'], resp_weight)
            self.route_props['mod_formats_list'].append(mod_format)
            self.route_props['paths_list'].append(path)

    def k_shortest_path(self, k_paths: int):
        """
        Finds the k-shortest-paths.

        :param k_paths: The number of paths to consider.
        """
        # This networkx function will always return the shortest paths in order
        paths_obj = nx.shortest_simple_paths(G=self.sdn_props['topology'], source=self.route_props['source'],
                                             target=self.route_props['destination'], weight='length')

        for k, path in enumerate(paths_obj):
            if k > k_paths - 1:
                break
            path_len = find_path_len(path, self.sdn_props['topology'])
            mod_format = get_path_mod(self.sdn_props['mod_formats'], path_len)

            self.route_props['paths_list'].append(path)
            self.route_props['mod_formats_list'].append([mod_format])
            self.route_props['weights_list'].append(path_len)

    def nli_aware(self):
        """
        Finds and selects the path with the least amount of non-linear impairment.
        """
        for link in self.sdn_props['net_spec_db']:
            source, destination = link[0], link[1]
            num_spans = self.sdn_props['topology'][source][destination]['length'] / self.route_props['span_len']

            link_cost = get_nli_cost(link=link, num_spans=num_spans, source=source,
                                     destination=destination)

            self.sdn_props['topology'][source][destination]['nli_cost'] = link_cost

        self.find_least_weight_path(weight='nli_cost')

    def xt_aware(self, beta: float, xt_type: str):
        """
        Calculates all path's costs with respect to intra-core cross-talk values and returns the path with the least
        amount of cross-talk interference.

        :param beta: A parameter used to determine the tradeoff between length and cross-talk value.
        :param xt_type: Whether we would like to consider length in the final calculation or not.
        :return: The path with the least amount of interference.
        :rtype: list
        """
        # At the moment, we have identical bidirectional links (no need to loop over all links)
        for link in list(self.sdn_props['net_spec_db'].keys())[::2]:
            source, destination = link[0], link[1]
            num_spans = self.sdn_props['topology'][source][destination]['length'] / self.route_props['span_len']

            free_slots = find_free_slots(net_spec_db=self.sdn_props['net_spec_db'], des_link=link)
            xt_cost = find_xt_link_cost(free_slots=free_slots, link_num=link)

            if xt_type == 'with_length':
                link_cost = (beta * (self.sdn_props['topology'][source][destination]['length'] /
                                     self.route_props['max_link'])) + ((1 - beta) * xt_cost)
            elif xt_type == 'without_length':
                link_cost = (num_spans / self.route_props['max_span']) * xt_cost
            else:
                raise ValueError(f'XT type not recognized, expected with or without_length, got: {xt_type}')

            self.sdn_props['topology'][source][destination]['xt_cost'] = link_cost
            self.sdn_props['topology'][destination][source]['xt_cost'] = link_cost

        self.find_least_weight_path(weight='xt_cost')
