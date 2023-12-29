import numpy as np
import copy
import json

from useful_functions.sim_functions import find_path_len
from useful_functions.handle_dirs_files import create_dir


# TODO: Do not re-create this each iteration, it's designed to work for multiple iters
# TODO: Double check for proper naming
# TODO: Potentially a copy of engine props here
# TODO: Check to see that all constructor vars are used, maybe add to dictionary while you do this
# TODO: Further organize this after integration
class SimStats:
    """
    The SimStats class finds and stores all relevant statistics in simulations.
    """

    def __init__(self, engine_props: dict, stats_props: dict = None):
        # TODO: To override or pickup simulation from an end point (not implemented yet)
        # TODO: A method to return this is in the correct format/save?
        self.stats_props = stats_props
        self.engine_props = engine_props

        # TODO: Add to standard, variable type if list or dict
        # tmp_obj
        self.bw_block_dict = dict()
        # request_snapshots
        self.snapshots_dict = dict()
        # active_requests
        # TODO: No longer needed
        self.active_reqs_dict = dict()
        # cores_chosen
        self.cores_dict = dict()
        # path_weights
        self.weights_dict = dict()
        # mods_used
        self.mods_used_dict = dict()
        self.block_bw_dict = dict()
        # TODO: Do not use this, something better
        self.stats_dict = {'block_per_sim': dict(), 'misc_stats': dict()}
        # block_reasons
        # TODO: Generalize maybe
        # TODO: Initialize after each iteration?
        self.block_reasons_dict = {'distance': None, 'congestion': None, 'xt_threshold': None}

        # TODO: Will incorporate num_trans by iteration number in the list list[iter_num] += 1
        # block_per_sim
        self.sim_block_list = list()
        # trans_arr
        self.trans_list = list()
        # hops
        self.hops_list = list()
        # path_lens
        self.lengths_list = list()
        # route_times
        self.route_times_list = list()

        # num_blocked_reqs
        # TODO: Update this in engine ++, and set to zero?
        self.num_trans = 0
        self.blocked_reqs = 0
        # blocking_mean
        self.block_mean = None
        # blocking_variance
        self.block_variance = None
        # block_ci
        # TODO: Will convert this, no need to have a ci and ci percent
        self.block_ci = None
        self.block_ci_percent = None

    # TODO: Add to standards doc (get)
    # TODO: Combine with get path slots
    # TODO: Change name to get snapshots dict or something
    # _get_total_occupied_slots
    # Also _get_path_free_slots
    def get_occupied_slots(self, net_spec_dict: dict, req_num: int, num_trans: int, path_list: list = None):
        """
        Finds the total number of occupied slots and guard bands currently allocated in the network or a specific path.

        :param net_spec_dict: The current network spectrum database.
        :param req_num: The current request number.
        :param num_trans: The number of transponders used for the request.
        :param path_list: The desired path to find the occupied slots on.
        :return: None
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
                requests = set(core[core > 0])
                for curr_req in requests:
                    active_reqs_set.add(curr_req)

                occupied_slots += len(np.where(core != 0)[0])
                guard_slots += len(np.where(core < 0)[0])

        self.snapshots_dict[req_num]['occupied_slots'].append(occupied_slots)
        self.snapshots_dict[req_num]['guard_slots'].append(guard_slots)
        self.snapshots_dict[req_num]['active_requests'].append(len(active_reqs_set))

        blocking_prob = self.blocked_reqs / req_num
        self.snapshots_dict[req_num]["blocking_prob"].append(blocking_prob)
        self.snapshots_dict[req_num]['num_segments'].append(num_trans)

    # TODO: Just pass an empty list of snapshot keys if you don't want them
    # TODO: Make sure to call init stats
    # TODO: Change cores_range to something else
    # TODO: Not the best name
    # TODO: Not all variables are being initialized here, hops, path lens, ect.
    def init_stats(self, num_requests: int, step: int, snap_keys_list: list, cores_range: range, mod_per_bw: dict):
        """
        Initializes data structures used in other methods of this class.

        :param num_requests: The number of requests for the simulation.
        :param step: How often to take snapshot statistics.
        :param snap_keys_list: Keys for desired stats to be tracked, for example, occupied slots.
        :param mod_per_bw: Contains the modulations and bandwidths used.
        :param cores_range: The range of core values used.
        :return: None
        """
        # TODO: Add step to configuration file
        for req_num in range(0, num_requests + 1, step):
            self.snapshots_dict[req_num] = dict()
            # TODO: Occupied slots, guard bands, blocking prob, num segments, active requests
            for key in snap_keys_list:
                self.snapshots_dict[req_num][key] = list()

        self.cores_dict = {key: 0 for key in cores_range}
        self.blocked_reqs = 0
        self.num_trans = 0

        # TODO: This desperately needs engine props in this class
        for bandwidth, obj in mod_per_bw.items():
            self.mods_used_dict[bandwidth] = dict()
            self.weights_dict[bandwidth] = dict()
            for modulation in obj.keys():
                self.weights_dict[bandwidth][modulation] = list()
                self.mods_used_dict[bandwidth][modulation] = 0

            self.block_bw_dict[bandwidth] = 0

        # TODO: This and all other variables need to be cheked for multiple iterations when saved
        self.block_reasons_dict = {'distance': 0, 'congestion': 0, 'xt_threshold': 0}

    # _calculate_block_percent
    # TODO: May move num_reqs to constructor
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

        self.sim_block_list.append(blocking_prob)

    # TODO: Implement this when you get that copy of engine props?
    def _get_cost_data(self, mod_format: str):
        # # TODO: Reset in init iter?? Check ALL variables
        # if self.engine_props['check_snr'] is None or self.engine_props['check_snr'] == 'None':
        #     self.path_weights[self.chosen_bw][path_mod].append(response_data['path_weight'])
        # else:
        #     self.path_weights[self.chosen_bw][path_mod].append(response_data['xt_cost'])
        pass
        # raise NotImplementedError

    # TODO: Move topology to constructor, will just need a copy of engine props
    # TODO: Not the best name, rename to update or something, this is not each iteration
    def get_iter_data(self, blocked: bool, req_data: dict, sdn_data: dict, topology):
        # TODO: Reset in init iter?? Check ALL variables
        if blocked:
            self.blocked_reqs += 1
            self.block_reasons_dict[sdn_data[1]] += 1
            self.block_bw_dict[req_data['bandwidth']] += 1
            # If the request was blocked, use one transponder
            self.num_trans += 1
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
            self.num_trans += num_transponders

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
            trans_mean = self.num_trans / num_reqs - self.blocked_reqs
            self.trans_list.append(trans_mean)

        if self.blocked_reqs > 0:
            for block_type, num_times in self.block_reasons_dict.items():
                # TODO: Should always be a float?
                self.block_reasons_dict[block_type] = num_times / float(self.blocked_reqs)

    # TODO: Add to constructor these params
    def print_iter_stats(self, max_iters: int, iteration: int, erlang: float):
        print(f"Iteration {iteration + 1} out of {max_iters} completed for Erlang: {erlang}")
        print(f'Mean of blocking: {np.mean(self.sim_block_list)}')
