import networkx as nx
import numpy as np

from arg_scripts.routing_args import empty_props
from helper_scripts.routing_helpers import get_nli_cost, find_xt_link_cost
from helper_scripts.sim_helpers import find_path_len, get_path_mod, find_free_slots


# TODO: Standardize return format
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

    def _find_least_cong(self):
        # Sort dictionary by number of free slots, descending
        sorted_paths_list = sorted(self.route_props['paths_list'], key=lambda d: d['link_info']['free_slots'],
                                   reverse=True)

        return sorted_paths_list[0]['path']

    def find_least_cong(self):
        """
        Find the least congested path in the network.

        :return: The least congested path.
        :rtype: list
        """
        all_paths_obj = nx.shortest_simple_paths(self.sdn_props['topology'], self.route_props['source'],
                                                 self.route_props['destination'])
        min_hops = None
        least_path_list = False

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
                    least_path_list = self._find_least_cong()

        self.route_props['paths_list'].append(least_path_list)
        # TODO: Constant QPSK format
        self.route_props['mod_formats_list'].append('QPSK')
        # TODO: Not sure what to put here
        self.route_props['weights_list'].append(None)

    def find_least_weight(self, weight: str):
        """
        Find the path with respect to a given weight.

        :param weight: Determines the weight to consider for finding the path.
        """
        paths_obj = nx.shortest_simple_paths(G=self.sdn_props['topology'], source=self.route_props['source'],
                                             target=self.route_props['destination'], weight=weight)

        for path_list in paths_obj:
            # If cross-talk, our path weight is the summation across the path
            if weight == 'xt_cost':
                resp_weight = sum(self.sdn_props['topology'][path_list[i]][path_list[i + 1]][weight]
                                  for i in range(len(path_list) - 1))
            else:
                resp_weight = find_path_len(path=path_list, topology=self.sdn_props['topology'])

            self.route_props['weights_list'].append(resp_weight)
            mod_format = get_path_mod(self.sdn_props['mod_formats'], resp_weight)
            self.route_props['mod_formats_list'].append(mod_format)
            self.route_props['paths_list'].append(path_list)

    def find_k_shortest(self):
        # This networkx function will always return the shortest paths in order
        paths_obj = nx.shortest_simple_paths(G=self.sdn_props['topology'], source=self.route_props['source'],
                                             target=self.route_props['destination'], weight='length')

        for k, path_list in enumerate(paths_obj):
            if k > self.engine_props['k_paths'] - 1:
                break
            path_len = find_path_len(path_list, self.sdn_props['topology'])
            mod_format = get_path_mod(self.sdn_props['mod_formats'], path_len)

            self.route_props['paths_list'].append(path_list)
            self.route_props['mod_formats_list'].append([mod_format])
            self.route_props['weights_list'].append(path_len)

    def find_least_nli(self):
        """
        Finds and selects the path with the least amount of non-linear impairment.
        """
        for link in self.sdn_props['net_spec_db']:
            source, destination = link[0], link[1]
            num_spans = self.sdn_props['topology'][source][destination]['length'] / self.route_props['span_len']

            link_cost = get_nli_cost(link=link, num_spans=num_spans, route_props=self.route_props,
                                     engine_props=self.engine_props)
            self.sdn_props['topology'][source][destination]['nli_cost'] = link_cost

        self.find_least_weight(weight='nli_cost')

    def find_least_xt(self):
        """
        Finds the path with the least amount of intra-core crosstalk interference.

        :return: The selected path with the least amount of interference.
        :rtype: list
        """
        # At the moment, we have identical bidirectional links (no need to loop over all links)
        for link_list in list(self.sdn_props['net_spec_db'].keys())[::2]:
            source, destination = link_list[0], link_list[1]
            num_spans = self.sdn_props['topology'][source][destination]['length'] / self.route_props['span_len']

            free_slots = find_free_slots(net_spec_db=self.sdn_props['net_spec_db'], des_link=link_list)
            xt_cost = find_xt_link_cost(free_slots=free_slots, link_num=link_list,
                                        net_spec_db=self.sdn_props['net_spec_db'])

            if self.engine_props['xt_type'] == 'with_length':
                link_cost = self.sdn_props['topology'][source][destination]['length'] / self.route_props['max_link']
                link_cost *= self.engine_props['beta']
                link_cost += (1 - self.engine_props['beta']) * xt_cost
            elif self.engine_props['xt_type'] == 'without_length':
                link_cost = (num_spans / self.route_props['max_span']) * xt_cost
            else:
                raise ValueError(f"XT type not recognized, expected with or without_length, "
                                 f"got: {self.engine_props['xt_type']}")

            self.sdn_props['topology'][source][destination]['xt_cost'] = link_cost
            self.sdn_props['topology'][destination][source]['xt_cost'] = link_cost

        self.find_least_weight(weight='xt_cost')

    def get_route(self, ai_obj: object):
        """
        Controls the class by finding the appropriate routing function.

        :param ai_obj: Artificial intelligence is handled in a separate class.
        :return: None
        """
        if self.engine_props['route_method'] == 'nli_aware':
            self.find_least_nli()
        elif self.engine_props['route_method'] == 'xt_aware':
            self.find_least_xt()
        elif self.engine_props['route_method'] == 'least_congested':
            self.find_least_cong()
        elif self.engine_props['route_method'] == 'shortest_path':
            self.find_least_weight(weight='length')
        elif self.engine_props['route_method'] == 'k_shortest_path':
            self.find_k_shortest()
        elif self.engine_props['route_method'] == 'ai':
            # TODO: Need to fix ai to account for passing props, probably don't need all three params
            path, mod_format = ai_obj.route(sdn_props=self.sdn_props, route_props=self.route_props)
            self.route_props['paths_list'] = [path]
            self.route_props['mod_formats_list'] = [mod_format]
        else:
            raise NotImplementedError(f"Routing method not recognized, got: {self.engine_props['route_method']}.")
