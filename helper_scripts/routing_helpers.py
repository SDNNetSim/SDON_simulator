import copy
import math

import numpy as np
import networkx as nx

from helper_scripts.sim_helpers import find_free_channels, find_taken_channels


class RoutingHelpers:
    """
    Class containing methods to assist with finding routes in the routing class.
    """

    def __init__(self, route_props: dict, engine_props: dict, sdn_props: dict):
        self.route_props = route_props
        self.engine_props = engine_props
        self.sdn_props = sdn_props

    def _get_simulated_link(self):
        sim_link_list = np.zeros(self.engine_props['spectral_slots'])
        # Add to the step to account for the guard band
        for i in range(0, len(sim_link_list), self.sdn_props['slots_needed'] + self.engine_props['guard_slots']):
            value_to_set = i // self.sdn_props['slots_needed'] + 1
            sim_link_list[i:i + self.sdn_props['slots_needed'] + 2] = value_to_set

        # Add guard-bands
        sim_link_list[self.sdn_props['slots_needed']::self.sdn_props['slots_needed'] +
                                                      self.sdn_props['guard_slots']] *= -1
        # Free the middle-most channel with respect to the number of slots needed
        center_index = len(sim_link_list) // 2
        if self.sdn_props['slots_needed'] % 2 == 0:
            start_index = center_index - self.sdn_props['slots_needed'] // 2
            end_idx = center_index + self.sdn_props['slots_needed'] // 2
        else:
            start_index = center_index - self.sdn_props['slots_needed'] // 2
            end_idx = center_index + self.sdn_props['slots_needed'] // 2 + 1

        sim_link_list[start_index:end_idx] = 0
        return sim_link_list

    def find_worst_nli(self, num_span: float):
        """
        Finds the worst possible non-linear impairment cost.

        :param num_span: The number of span a link has.
        :return: The worst NLI.
        :rtype: float
        """
        links_list = list(self.sdn_props['net_spec_db'].keys())
        sim_link_list = self._get_simulated_link()

        orig_link_list = copy.copy(self.sdn_props['net_spec_db'][links_list[0]]['cores_matrix'][0])
        self.sdn_props['net_spec_db'][links_list[0]]['cores_matrix'][0] = sim_link_list

        free_channels_dict = find_free_channels(net_spec_db=self.sdn_props['net_spec_db'],
                                                slots_needed=self.sdn_props['slots_needed'], des_link=links_list[0])
        taken_channels_dict = find_taken_channels(net_spec_db=self.sdn_props['net_spec_db'], des_link=links_list[0])

        nli_worst = self._find_link_cost(free_channels_dict=free_channels_dict, taken_channels_dict=taken_channels_dict,
                                         num_span=num_span)
        self.sdn_props['net_spec_db'][links_list[0]]['cores_matrix'][0] = orig_link_list
        return nli_worst

    def _find_link_cost(self, free_channels_dict: dict, taken_channels_dict: dict, num_span: float):
        nli_cost = 0
        num_channels = 0
        for core_num, free_channels in free_channels_dict.items():
            # Update MCI for available channel
            for channel in free_channels:
                num_channels += 1
                # Calculate the center frequency for the open channel
                center_freq = (channel[0] * self.route_props['freq_spacing']) + ((self.sdn_props['slots_needed'] *
                                                                                  self.route_props['freq_spacing']) / 2)

                nli_cost += self._find_channel_mci(channels_list=taken_channels_dict[core_num], center_freq=center_freq,
                                                   num_span=num_span)

        # A constant score of 1000 if the link is fully congested
        if num_channels == 0:
            return 1000.0

        link_cost = nli_cost / num_channels
        return link_cost

    def _find_channel_mci(self, channels_list: list, center_freq: float, num_span: float):
        mci = 0
        for channel in channels_list:
            # The current center frequency for the occupied channel
            curr_freq = (channel[0] * self.route_props['freq_spacing']) + \
                        ((len(channel) * self.route_props['freq_spacing']) / 2)
            bandwidth = len(channel) * self.route_props['freq_spacing']
            # Power spectral density
            power_spec_dens = self.route_props['input_power'] / bandwidth

            mci += (power_spec_dens ** 2) * math.log(abs((abs(center_freq - curr_freq) + (bandwidth / 2)) / (
                    abs(center_freq - curr_freq) - (bandwidth / 2))))

        mci = (mci / self.route_props['mci_worst']) * num_span
        return mci

    @staticmethod
    def _find_adjacent_cores(core_num: int):
        # Identify the adjacent cores to the currently selected core
        # The neighboring core directly before the currently selected core
        before = 5 if core_num == 0 else core_num - 1
        # The neighboring core directly after the currently selected core
        after = 0 if core_num == 5 else core_num + 1

        return before, after

    def _find_num_overlapped(self, channel: int, core_num: int, link_num: int, net_spec_db: dict):
        num_overlapped = 0.0
        if core_num != 6:
            adj_cores_tuple = self._find_adjacent_cores(core_num=core_num)
            if net_spec_db[link_num]['cores_matrix'][adj_cores_tuple[0]][channel] > 0:
                num_overlapped += 1
            if net_spec_db[link_num]['cores_matrix'][adj_cores_tuple[1]][channel] > 0:
                num_overlapped += 1
            if net_spec_db[link_num]['cores_matrix'][6][channel] > 0:
                num_overlapped += 1

            num_overlapped /= 3
        # The number of overlapped cores for core six will be different since it's the center core
        else:
            for sub_core_num in range(6):
                if net_spec_db[link_num]['cores_matrix'][sub_core_num][channel] > 0:
                    num_overlapped += 1

            num_overlapped /= 6

        return num_overlapped

    # fixme: Only works for seven cores
    # TODO: Finish docstring, not sure what free_slots_dict is
    def find_xt_link_cost(self, free_slots_dict: dict, link_num: int):
        """
        Finds the intra-core crosstalk cost for a single link.

        :param free_slots_dict:
        :param link_num:
        :return:
        """
        xt_cost = 0
        num_free_slots = 0
        for core_num in free_slots_dict:
            num_free_slots += len(free_slots_dict[core_num])
            for channel in free_slots_dict[core_num]:
                num_overlapped = self._find_num_overlapped(channel=channel, core_num=core_num, link_num=link_num,
                                                           net_spec_db=self.sdn_props['net_spec_db'])
                xt_cost += num_overlapped

        # A constant score of 1000 if the link is fully congested
        if num_free_slots == 0:
            return 1000.0

        link_cost = xt_cost / num_free_slots
        return link_cost

    def get_nli_path(self, path_list: list):
        """
        Find the non-linear impairment for a single path.

        :param path_list: The given path.
        :return: The NLI calculation for the path.
        :rtype: float
        """
        nli_cost = 0
        for source, destination in zip(path_list, path_list[1:]):
            num_span = self.engine_props['topology'][source][destination]['length'] / self.route_props['span_len']
            link_tuple = (source, destination)
            nli_cost += self.get_nli_cost(link_tuple=link_tuple, num_span=num_span)

        return nli_cost

    def get_nli_cost(self, link_list: tuple, num_span: float):
        free_channels_dict = find_free_channels(net_spec_db=self.sdn_props['net_spec_dict'],
                                                slots_needed=self.sdn_props['slots_needed'], des_link=link_list)
        taken_channels_dict = find_taken_channels(net_spec_db=self.sdn_props['net_spec_dict'], des_link=link_list)

        nli_cost = self._find_link_cost(free_channels_dict=free_channels_dict, taken_channels_dict=taken_channels_dict,
                                        num_span=num_span)
        # Tradeoff between link length and the non-linear impairment cost
        source, destination = link_list[0], link_list[1]

        if self.route_props['max_link_length'] is None:
            topology = self.engine_props['topology']
            self.route_props['max_link_length'] = max(nx.get_edge_attributes(topology, 'length').values(), default=0.0)

        nli_cost = (self.engine_props['beta'] *
                    (self.engine_props['topology'][source][destination]['length'] /
                     self.route_props['max_link_length'])) + ((1 - self.engine_props['beta']) * nli_cost)

        return nli_cost
