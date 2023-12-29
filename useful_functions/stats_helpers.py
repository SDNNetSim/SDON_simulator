import copy
import json

import numpy as np

from stats_args import empty_props
from stats_args import SNAP_KEYS_LIST
from useful_functions.sim_functions import find_path_len
from useful_functions.handle_dirs_files import create_dir


# TODO: Add SIGINT and SIGTERM
# TODO: Review code:
#   - Read doc strings
#   - Variable names
#   - Code readability/efficiency
#   - Move repeat variables to self
#   - Use constant variables from engine props!
class SimStats:
    """
    The SimStats class finds and stores all relevant statistics in simulations.
    """

    def __init__(self, engine_props: dict, stats_props: dict = None):
        # TODO: Implement ability to pick up from previously run simulations
        if stats_props is not None:
            self.stats_props = stats_props
        else:
            self.stats_props = empty_props

        self.engine_props = engine_props

        # Used to track transponders for a single allocated request
        self.curr_trans = 0
        # Used to track transponders for an entire simulation on average
        self.total_trans = 0
        self.blocked_reqs = 0
        self.block_mean = None
        self.block_variance = None
        self.block_ci = None
        self.block_ci_percent = None

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

    # TODO: We never actually call path list here
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

    def _init_snapshots(self, num_reqs: int, snapshot_step: int):
        for req_num in range(0, num_reqs + 1, snapshot_step):
            self.stats_props['snapshots_dict'][req_num] = dict()
            for key in SNAP_KEYS_LIST:
                self.stats_props['snapshots_dict'][req_num][key] = list()

    def _init_mods_weights_bws(self, mod_per_bw: dict):
        for bandwidth, obj in mod_per_bw.items():
            self.stats_props['mods_used_dict'][bandwidth] = dict()
            self.stats_props['weights_dict'][bandwidth] = dict()
            for modulation in obj.keys():
                self.stats_props['weights_dict'][bandwidth][modulation] = list()
                self.stats_props['mods_used_dict'][bandwidth][modulation] = 0

            self.stats_props['block_bw_dict'][bandwidth] = 0

    # TODO: Take advantage of engine props for sure
    def _init_stat_dicts(self, mod_per_bw: dict, num_reqs: int, snapshot_step: int, core_range: range):
        for stat_key in self.stats_props:
            if stat_key == 'mods_used_dict' or stat_key == 'weights_dict' or stat_key == 'block_bw_dict':
                self._init_mods_weights_bws(mod_per_bw=mod_per_bw)
            elif stat_key == 'snapshots_dict':
                self._init_snapshots(num_reqs=num_reqs, snapshot_step=snapshot_step)
            elif stat_key == 'cores_dict':
                self.stats_props['cores_dict'] = {key: 0 for key in core_range}
            elif stat_key == 'block_reasons_dict':
                self.stats_props['block_reasons_dict'] = {'distance': 0, 'congestion': 0, 'xt_threshold': 0}
            else:
                raise ValueError('Dictionary statistic was not reset in props.')

    def _init_stat_lists(self):
        for stat_key in self.stats_props:
            if isinstance(self.stats_props[stat_key], list):
                self.stats_props[stat_key] = list()

    # TODO: Not all variables are being initialized here, hops, path lens, etc.
    def init_iter_stats(self, num_reqs: int, snapshot_step: int, core_range: range, mod_per_bw: dict):
        """
        Initializes data structures used in other methods of this class.

        :param num_reqs: The number of requests for the simulation.
        :param snapshot_step: How often to take snapshot statistics.
        :param mod_per_bw: Contains the modulations and bandwidths used.
        :param core_range: The range of core values used.
        :return: None
        """
        self._init_stat_dicts(num_reqs=num_reqs, snapshot_step=snapshot_step, core_range=core_range,
                              mod_per_bw=mod_per_bw)
        # TODO: A function to loop these and set them to zero?
        self.blocked_reqs = 0
        self.total_trans = 0

    def get_blocking(self, num_reqs: int):
        """
        Gets the current blocking probability.

        :param num_reqs: The total number of requests in the simulation.
        :return: None
        """
        if num_reqs == 0:
            blocking_prob = 0
        else:
            blocking_prob = self.blocked_reqs / num_reqs
        self.stats_props['sim_block_list'].append(blocking_prob)

    # TODO: Make sure this is called in engine
    def _get_cost_data(self, mod_format: str):
        if self.engine_props['check_snr'] is None or self.engine_props['check_snr'] == 'None':
            self.path_weights[self.chosen_bw][path_mod].append(response_data['path_weight'])
        else:
            self.path_weights[self.chosen_bw][path_mod].append(response_data['xt_cost'])

    # TODO: Move topology to constructor, will just need a copy of engine props
    # TODO: Not the best name, rename to update or something, this is not each iteration
    # TODO: Change resp
    def get_iter_data(self, req_data: dict, sdn_data: dict, topology):
        # TODO: Reset in init iter?? Check ALL variables
        if not sdn_data[0]:
            self.blocked_reqs += 1
            self.block_reasons_dict[sdn_data[1]] += 1
            self.block_bw_dict[req_data['bandwidth']] += 1
            # If the request was blocked, use one transponder
            self.total_trans += 1
        else:
            # response data is sdn_data[0] and num transponders is resp[2]
            num_hops = len(sdn_data[0]['path']) - 1
            self.hops_list.append(num_hops)

            # TODO: Fix
            path_len = find_path_len(path=sdn_data[0]['path'], topology=topology)
            self.lengths_list.append(path_len)

            core_chosen = sdn_data[0]['spectrum']['core_num']
            self.cores_dict[core_chosen] += 1

            self.route_times_list.append(sdn_data[0]['route_time'])

            mod_format = sdn_data[0]['mod_format']
            bandwidth = req_data['bandwidth']
            self.mods_used_dict[bandwidth][mod_format] += 1

            num_transponders = sdn_data[2]
            # TODO: Use this for the get occupied slots function
            self.total_trans += num_transponders

            # TODO: Implement this when you get that copy of engine props?
            self._get_cost_data(mod_format='None')

    # TODO: This comes from stats_dict in engine, the 'block_per_sim' key
    # TODO: Need to call this in engine
    # TODO: Double check type of erlang
    def get_conf_inter(self, iteration: int, erlang: float):
        """
        Get the confidence interval for every iteration so far.

        :param iteration: The iteration number.
        :param erlang: The current traffic volume (erlang).
        :return: Whether the simulations should end for this erlang.
        :rtype: bool
        """
        blocking_mean = np.mean(self.sim_block_list)
        if blocking_mean == 0.0 or len(self.sim_block_list) <= 1:
            return False

        blocking_variance = np.var(self.sim_block_list)
        try:
            # TODO: Change?
            block_ci_rate = 1.645 * (np.sqrt(blocking_variance) / np.sqrt(len(self.sim_block_list)))
            self.block_ci = block_ci_rate
            block_ci_percent = ((2 * block_ci_rate) / blocking_mean) * 100
            self.block_ci_percent = block_ci_percent
        except ZeroDivisionError:
            return False

        # TODO: Add to configuration file (ci percent)
        # TODO: Need to save stats somehow? Add file_type to config file and put it in props here
        # TODO: Needs to return true and false?
        if block_ci_percent <= 5:
            print(f"Confidence interval of {round(block_ci_percent, 2)}% reached. "
                  f"{iteration + 1}, ending and saving results for Erlang: {erlang}")
            self.save_stats(iteration=iteration)
            return True

        return False

    # TODO: Still need to save here
    # TODO: Need to return here?
    # TODO: Change default file type saving, will be put in config file
    # TODO: This needs to be called more often, not just on termination
    def save_stats(self, iteration: int, sim_info: str, file_type: str = 'json'):
        """
        Saves simulations stats as either a json or csv file.

        :param file_type: The desired file type, either json or csv.
        :param iteration: The iteration that is about to finish.
        :param sim_info: Contains the topology, date, and time used for saving the file.
        :return: None
        """
        # TODO: Still need to save as either json or csv file format, methods to help out with that
        if file_type not in ('json', 'csv'):
            raise NotImplementedError(f'Invalid file type: {file_type}, expected csv or json.')

        for _, curr_snapshot in self.snapshots_dict.items():
            for snap_key, data_list in curr_snapshot.items():
                curr_snapshot[snap_key] = np.mean(data_list)

        for _, mod_obj in self.weights_dict.items():
            for modulation, lst in mod_obj.items():
                # Modulation was never used
                if len(lst) == 0:
                    mod_obj[modulation] = {'mean': None, 'std': None, 'min': None, 'max': None}
                else:
                    mod_obj[modulation] = {'mean': float(np.mean(lst)), 'std': float(np.std(lst)),
                                           'min': float(np.min(lst)), 'max': float(np.max(lst))}

        # TODO: Do not use this, something way better
        self.stats_dict['blocking_mean'] = self.block_mean
        self.stats_dict['blocking_variance'] = self.block_variance
        self.stats_dict['ci_rate_block'] = self.block_ci
        self.stats_dict['ci_percent_block'] = self.block_ci_percent

        # TODO: Add this after we get engine props
        # if self.iteration == 0:
        #     self.stats_dict['sim_params'] = self.engine_props

        # TODO: Ridiculous, make this better, suggestions above
        self.stats_dict['misc_stats'][iteration] = {
            'trans_mean': float(np.mean(self.trans_list)),
            'block_reasons': self.block_reasons_dict,
            'block_per_bw': {key: float(np.mean(lst)) for key, lst in self.block_bw_dict.items()},
            'request_snapshots': self.snapshots_dict,
            'hops': {'mean': float(np.mean(self.hops_list)), 'min': float(np.min(self.hops_list)),
                     'max': float(np.max(self.hops_list))},
            'route_times': float(np.mean(self.route_times_list)),
            # TODO: Use python built in functions
            'path_lengths': {'mean': float(np.mean(self.lengths_list)), 'min': float(np.min(self.lengths_list)),
                             'max': float(np.max(self.lengths_list))},
            'cores_chosen': self.cores_dict,
            'weight_info': self.weights_dict,
            'modulation_formats': self.mods_used_dict,
        }

        # TODO: Use OS module
        base_fp = "data/output/"

        # TODO: Keep this in engine most likely
        # if self.engine_props['route_method'] == 'ai':
        #     self.ai_obj.save()

        # Save threads to child directories
        # TODO: Use OS module
        # TODO: Integrate this somehow, through engine props
        base_fp += f"/{sim_info}/{self.engine_props['thread_num']}"
        create_dir(base_fp)

        # TODO: Comment
        tmp_topology = copy.deepcopy(self.engine_props['topology'])
        del self.engine_props['topology']
        with open(f"{base_fp}/{self.engine_props['erlang']}_erlang.json", 'w', encoding='utf-8') as file_path:
            json.dump(self.stats_dict, file_path, indent=4)

        self.engine_props['topology'] = tmp_topology

    # TODO: Will rename to update iter stats or something
    def end_iter_stats(self, num_reqs: int):
        # TODO: This won't work, trans_list keeps track of all transponders used in an iteration
        if num_reqs == self.blocked_reqs:
            self.trans_list.append(0)
        else:
            # TODO: This SHOULD work but needs to be checked
            trans_mean = self.total_trans / num_reqs - self.blocked_reqs
            self.trans_list.append(trans_mean)

        if self.blocked_reqs > 0:
            for block_type, num_times in self.block_reasons_dict.items():
                # TODO: Should always be a float?
                self.block_reasons_dict[block_type] = num_times / float(self.blocked_reqs)

    # TODO: Add to constructor these params
    def print_iter_stats(self, max_iters: int, iteration: int, erlang: float):
        print(f"Iteration {iteration + 1} out of {max_iters} completed for Erlang: {erlang}")
        print(f'Mean of blocking: {np.mean(self.sim_block_list)}')
