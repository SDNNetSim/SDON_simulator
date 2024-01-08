import json
import os
import math
import copy
from statistics import mean, variance, stdev

import numpy as np

from arg_scripts.stats_args import empty_props
from arg_scripts.stats_args import SNAP_KEYS_LIST
from helper_scripts.sim_helpers import find_path_len
from helper_scripts.os_helpers import create_dir


class SimStats:
    """
    The SimStats class finds and stores all relevant statistics in simulations.
    """

    def __init__(self, engine_props: dict, sim_info: str, stats_props: dict = None):
        # TODO: Implement ability to pick up from previously run simulations
        if stats_props is not None:
            self.stats_props = stats_props
        else:
            self.stats_props = empty_props

        self.engine_props = engine_props
        self.sim_info = sim_info

        self.save_dict = {'iter_stats': {}}

        # Used to track transponders for a single allocated request
        self.curr_trans = 0
        # Used to track transponders for an entire simulation on average
        self.total_trans = 0
        self.blocked_reqs = 0
        self.block_mean = None
        self.block_variance = None
        self.block_ci = None
        self.block_ci_percent = None
        self.topology = None
        self.iteration = None

    @staticmethod
    def _get_snapshot_info(net_spec_dict: dict, path_list: list):
        """
        Retrieves relative information for simulation snapshots.

        :param net_spec_dict: The current network spectrum database.
        :param path_list: A path to find snapshot info, if empty, does this for the entire network.
        :return: The occupied slots, number of guard bands, and active requests.
        :rtype: tuple
        """
        active_reqs_set = set()
        occupied_slots = 0
        guard_slots = 0
        # Skip by two because the link is bidirectional, no need to check both arrays e.g., (0, 1) and (1, 0)
        for link in list(net_spec_dict.keys())[::2]:
            if path_list is not None and link not in path_list:
                continue
            link_data = net_spec_dict[link]
            for core in link_data['cores_matrix']:
                requests_set = set(core[core > 0])
                for curr_req in requests_set:
                    active_reqs_set.add(curr_req)
                occupied_slots += len(np.where(core != 0)[0])
                guard_slots += len(np.where(core < 0)[0])

        return occupied_slots, guard_slots, len(active_reqs_set)

    def update_snapshot(self, net_spec_dict: dict, req_num: int, path_list: list = None):
        """
        Finds the total number of occupied slots and guard bands currently allocated in the network or a specific path.

        :param net_spec_dict: The current network spectrum database.
        :param req_num: The current request number.
        :param path_list: The desired path to find the occupied slots on.
        :return: None
        """
        occupied_slots, guard_slots, active_reqs = self._get_snapshot_info(net_spec_dict=net_spec_dict,
                                                                           path_list=path_list)
        blocking_prob = self.blocked_reqs / req_num

        self.stats_props['snapshots_dict'][req_num]['occupied_slots'].append(occupied_slots)
        self.stats_props['snapshots_dict'][req_num]['guard_slots'].append(guard_slots)
        self.stats_props['snapshots_dict'][req_num]['active_requests'].append(active_reqs)
        self.stats_props['snapshots_dict'][req_num]["blocking_prob"].append(blocking_prob)
        self.stats_props['snapshots_dict'][req_num]['num_segments'].append(self.curr_trans)

    def _init_snapshots(self):
        for req_num in range(0, self.engine_props['num_requests'] + 1, self.engine_props['snapshot_step']):
            self.stats_props['snapshots_dict'][req_num] = dict()
            for key in SNAP_KEYS_LIST:
                self.stats_props['snapshots_dict'][req_num][key] = list()

    def _init_mods_weights_bws(self):
        for bandwidth, obj in self.engine_props['mod_per_bw'].items():
            self.stats_props['mods_used_dict'][bandwidth] = dict()
            self.stats_props['weights_dict'][bandwidth] = dict()
            for modulation in obj.keys():
                self.stats_props['weights_dict'][bandwidth][modulation] = list()
                self.stats_props['mods_used_dict'][bandwidth][modulation] = 0

            self.stats_props['block_bw_dict'][bandwidth] = 0

    def _init_stat_dicts(self):
        for stat_key, data_type in self.stats_props.items():
            if not isinstance(data_type, dict):
                continue
            if stat_key in ('mods_used_dict', 'weights_dict', 'block_bw_dict'):
                self._init_mods_weights_bws()
            elif stat_key == 'snapshots_dict':
                self._init_snapshots()
            elif stat_key == 'cores_dict':
                self.stats_props['cores_dict'] = {core: 0 for core in range(self.engine_props['cores_per_link'])}
            elif stat_key == 'block_reasons_dict':
                self.stats_props['block_reasons_dict'] = {'distance': 0, 'congestion': 0, 'xt_threshold': 0}
            elif stat_key != 'iter_stats':
                raise ValueError('Dictionary statistic was not reset in props.')

    def _init_stat_lists(self):
        for stat_key in self.stats_props:
            if isinstance(self.stats_props[stat_key], list):
                # Only reset sim_block_list when we encounter a new traffic volume
                if self.iteration != 0 and stat_key == 'sim_block_list':
                    continue
                self.stats_props[stat_key] = list()

    def init_iter_stats(self):
        """
        Initializes data structures used in other methods of this class.

        :return: None
        """
        self._init_stat_dicts()
        self._init_stat_lists()

        self.blocked_reqs = 0
        self.total_trans = 0

    def get_blocking(self):
        """
        Gets the current blocking probability.

        :return: None
        """
        if self.engine_props['num_requests'] == 0:
            blocking_prob = 0
        else:
            blocking_prob = self.blocked_reqs / self.engine_props['num_requests']

        self.stats_props['sim_block_list'].append(blocking_prob)

    def iter_update(self, req_data: dict, sdn_data: dict):
        """
        Continuously updates the statistical data for each request allocated/blocked in the current iteration.

        :param req_data: Holds data relevant to the current request.
        :param sdn_data: Holds the response data from the sdn controller.
        :return: None
        """
        # Request was blocked
        if not sdn_data[0]:
            self.blocked_reqs += 1
            self.stats_props['block_reasons_dict'][sdn_data[1]] += 1
            self.stats_props['block_bw_dict'][req_data['bandwidth']] += 1
        else:
            num_hops = len(sdn_data[0]['path']) - 1
            self.stats_props['hops_list'].append(num_hops)

            path_len = find_path_len(path=sdn_data[0]['path'], topology=self.topology)
            self.stats_props['lengths_list'].append(path_len)

            core_chosen = sdn_data[0]['spectrum']['core_num']
            self.stats_props['cores_dict'][core_chosen] += 1

            self.stats_props['route_times_list'].append(sdn_data[0]['route_time'])

            mod_format = sdn_data[0]['mod_format']
            bandwidth = req_data['bandwidth']
            self.stats_props['mods_used_dict'][bandwidth][mod_format] += 1

            self.total_trans += sdn_data[2]
            # TODO: This won't work but after changing the sdn_controller it will (standardize path weight return)
            self.stats_props['weights_dict'][bandwidth][mod_format].append(sdn_data[0]['path_weight'])

    def _get_iter_means(self):
        for _, curr_snapshot in self.stats_props['snapshots_dict'].items():
            for snap_key, data_list in curr_snapshot.items():
                if data_list:
                    curr_snapshot[snap_key] = mean(data_list)
                else:
                    curr_snapshot[snap_key] = None

        for _, mod_obj in self.stats_props['weights_dict'].items():
            for modulation, data_list in mod_obj.items():
                # Modulation was never used
                if len(data_list) == 0:
                    mod_obj[modulation] = {'mean': None, 'std': None, 'min': None, 'max': None}
                else:
                    mod_obj[modulation] = {'mean': mean(data_list), 'std': stdev(data_list),
                                           'min': min(data_list), 'max': max(data_list)}

    def end_iter_update(self):
        """
        Updates relevant stats after an iteration has finished.

        :return: None
        """
        if self.engine_props['num_requests'] == self.blocked_reqs:
            self.stats_props['trans_list'].append(0)
        else:
            trans_mean = self.total_trans / float(self.engine_props['num_requests'] - self.blocked_reqs)
            self.stats_props['trans_list'].append(trans_mean)

        if self.blocked_reqs > 0:
            for block_type, num_times in self.stats_props['block_reasons_dict'].items():
                self.stats_props['block_reasons_dict'][block_type] = num_times / float(self.blocked_reqs)

        self._get_iter_means()

    def get_conf_inter(self):
        """
        Get the confidence interval for every iteration so far.

        :return: Whether the simulations should end for this erlang.
        :rtype: bool
        """
        self.block_mean = mean(self.stats_props['sim_block_list'])
        if self.block_mean == 0.0 or len(self.stats_props['sim_block_list']) <= 1:
            return False

        blocking_variance = variance(self.stats_props['sim_block_list'])
        try:
            # TODO: The desired CI rate should be in the configuration file (Ask Arash what they are again)
            block_ci_rate = 1.645 * (math.sqrt(blocking_variance) / math.sqrt(len(self.stats_props['sim_block_list'])))
            self.block_ci = block_ci_rate
            block_ci_percent = ((2 * block_ci_rate) / self.block_mean) * 100
            self.block_ci_percent = block_ci_percent
        except ZeroDivisionError:
            return False

        # TODO: Add to configuration file (ci percent, same as above)
        if block_ci_percent <= 5:
            print(f"Confidence interval of {round(block_ci_percent, 2)}% reached. "
                  f"{self.iteration + 1}, ending and saving results for Erlang: {self.engine_props['erlang']}")
            self.save_stats()
            return True

        return False

    # TODO: Implement batch saves
    def save_stats(self):
        """
        Saves simulations stats as either a json or csv file.

        :return: None
        """
        if self.engine_props['file_type'] not in ('json', 'csv'):
            raise NotImplementedError(f"Invalid file type: {self.engine_props['file_type']}, expected csv or json.")

        self.save_dict['blocking_mean'] = self.block_mean
        self.save_dict['blocking_variance'] = self.block_variance
        self.save_dict['ci_rate_block'] = self.block_ci
        self.save_dict['ci_percent_block'] = self.block_ci_percent

        self.save_dict['iter_stats'][self.iteration] = dict()
        for stat_key in self.stats_props:
            if stat_key in ('trans_list', 'hops_list', 'lengths_list', 'route_times_list'):
                mean_key = f"{stat_key.split('list')[0]}mean"
                self.save_dict['iter_stats'][self.iteration][mean_key] = mean(self.stats_props[stat_key])
            else:
                self.save_dict['iter_stats'][self.iteration][stat_key] = copy.deepcopy(self.stats_props[stat_key])

        save_fp = os.path.join('data', 'output', self.sim_info, self.engine_props['thread_num'])
        create_dir(save_fp)
        if self.engine_props['file_type'] == 'json':
            with open(f"{save_fp}/{self.engine_props['erlang']}_erlang.json", 'w', encoding='utf-8') as file_path:
                json.dump(self.save_dict, file_path, indent=4)
        else:
            raise NotImplementedError

    def print_iter_stats(self, max_iters: int):
        """
        Prints iteration stats, mostly used to ensure simulations are running fine.

        :param max_iters: The maximum number of iterations.
        :return: None
        """
        print(f"Iteration {self.iteration + 1} out of {max_iters} completed for Erlang: {self.engine_props['erlang']}")
        print(f"Mean of blocking: {round(mean(self.stats_props['sim_block_list']), 4)}")
