import numpy as np


# TODO: Do not re-create this each iteration, it's designed to work for multiple iters
# TODO: Double check for proper naming
class SimStats:
    """
    The SimStats class finds and stores all relevant statistics in simulations.
    """

    def __init__(self, stats_props: dict = None):
        # TODO: To override or pickup simulation from an end point (not implemented yet)
        # TODO: A method to return this is in the correct format/save?
        self.stats_props = stats_props

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
        # block_reasons
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
        self.blocked_reqs = 0
        # blocking_mean
        self.block_mean = None
        # blocking_variance
        self.block_variance = None
        # block_ci
        # TODO: Will convert this, no need to have a ci and ci percent
        self.block_ci = None

    # TODO: Add to standards doc (get)
    # TODO: Combine with get path slots
    # _get_total_occupied_slots
    # Also _get_path_free_slots
    def get_occupied_slots(self, net_spec_dict: dict, req_num: int, path_list: list = None):
        """
        Finds the total number of occupied slots and guard bands currently allocated in the network or a specific path.

        :param net_spec_dict: The current network spectrum database.
        :param req_num: The current request number.
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

    # TODO: Just pass an empty list of snapshot keys if you don't want them
    # TODO: Make sure to call init stats
    def init_stats(self, num_requests: int, step: int, snap_keys_list: list):
        """
        Initializes data structures used in other methods of this class.

        :param num_requests: The number of requests for the simulation.
        :param step: How often to take snapshot statistics.
        :param snap_keys_list: Keys for desired stats to be tracked, for example, occupied slots.
        :return: None
        """
        # TODO: Add step to configuration file
        for req_num in range(0, num_requests + 1, step):
            self.snapshots_dict[req_num] = dict()
            # TODO: Occupied slots, guard bands, blocking prob, num segments, active requests
            for key in snap_keys_list:
                self.snapshots_dict[req_num][key] = list()

        self.blocked_reqs = 0

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
            block_ci_rate = 1.645 * (np.sqrt(blocking_variance) / np.sqrt(len(self.sim_block_list)))
            block_ci_percent = ((2 * block_ci_rate) / blocking_mean) * 100
        except ZeroDivisionError:
            return False

        # TODO: Add to configuration file (ci percent)
        # TODO: Need to save stats somehow? Add file_type to config file and put it in props here
        # TODO: Needs to return true and false?
        if block_ci_percent <= 5:
            print(f"Confidence interval of {round(block_ci_percent, 2)}% reached. "
                  f"{iteration + 1}, ending and saving results for Erlang: {erlang}")
            self.save_stats()
            return True

        return False

    # TODO: Still need to save here
    # TODO: Need to return here?
    # TODO: Change default file type saving, will be put in config file
    # TODO: This needs to be called more often, not just on termination
    def save_stats(self, file_type: str = 'json'):
        """
        Saves simulations stats as either a json or csv file.

        :param file_type: The desired file type, either json or csv.
        :return: None
        """
        if file_type not in ('json', 'csv'):
            raise NotImplementedError(f'Invalid file type: {file_type}, expected csv or json.')

        for _, curr_snapshot in self.snapshots_dict.items():
            for snap_key, data_list in curr_snapshot.items():
                curr_snapshot[snap_key] = np.mean(data_list)

    # TODO: Add to constructor these params
    def print_iter_stats(self, max_iters: int, iteration: int, erlang: float):
        print(f"Iteration {iteration + 1} out of {max_iters} completed for Erlang: {erlang}")
        print(f'Mean of blocking: {np.mean(self.sim_block_list)}')
