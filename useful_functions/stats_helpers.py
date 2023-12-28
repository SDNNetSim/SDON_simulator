import numpy as np


# TODO: Do not re-create this each iteration, it's designed to work for multiple iters
class SimStats:
    """
    The SimStats class finds and stores all relevant statistics in simulations.
    """

    def __init__(self, stats_props: dict = None):
        # TODO: To override or pickup simulation from an end point (not implemented yet)
        # TODO: A method to return this is in the correct format/save?
        self.stats_props = stats_props

        # TODO: Add to standard, variable type if list or dict
        # block_per_sim
        self.sim_block_dict = dict()
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
        # trans_arr
        self.trans_list = list()
        # hops
        self.hops_list = list()
        # path_lens
        self.lengths_list = list()
        # route_times
        self.route_times_list = list()

        # num_blocked_reqs
        self.num_blocked_reqs = None
        # blocking_mean
        self.block_mean = None
        # blocking_variance
        self.block_variance = None
        # block_ci
        # TODO: Will convert this, no need to have a ci and ci percent
        self.block_ci = None

    # TODO: Add to standards doc (get)
    # _get_total_occupied_slots
    def get_occupied_slots(self, net_spec_db: dict, req_num: int):
        """
        Finds the total number of occupied slots and guard bands currently allocated in the network.

        :param net_spec_db: The current network spectrum database.
        :param req_num: The current request number.
        :return: None
        """
        active_reqs_set = set()
        occupied_slots = 0
        guard_slots = 0

        # Skip by two because the link is bidirectional, no need to check both arrays e.g., (0, 1) and (1, 0)
        for link in list(net_spec_db.keys())[::2]:
            link_data = net_spec_db[link]
            for core in link_data['cores_matrix']:
                requests = set(core[core > 0])
                for curr_req in requests:
                    active_reqs_set.add(curr_req)

                occupied_slots += len(np.where(core != 0)[0])
                guard_slots += len(np.where(core < 0)[0])

        try:
            self.snapshots_dict[req_num]['occupied_slots'].append(occupied_slots)
            self.snapshots_dict[req_num]['guard_slots'].append(guard_slots)
            self.snapshots_dict[req_num]['active_requests'].append(len(active_reqs_set))
        except KeyError as exc:
            raise KeyError(f"Please add occupied_slots, guard_slots, and active_requests as keys to use this method. "
                           f"Snapshots dict only has: {list(self.snapshots_dict[0].keys())}") from exc

    # TODO: Just pass an empty list of snapshot keys if you don't want them
    # TODO: Make sure to call init stats
    def init_stats(self, num_requests: int, step: int, snapshot_keys: list):
        """
        Initializes data structures used in other methods of this class.

        :param num_requests: The number of requests for the simulation.
        :param step: How often to take snapshot statistics.
        :param snapshot_keys: Keys for desired stats to be tracked, for example, occupied slots.
        :return: None
        """
        # TODO: Add step to configuration file
        for req_num in range(0, num_requests + 1, step):
            self.snapshots_dict[req_num] = dict()
            # TODO: Occupied slots, guard bands, blocking prob, num segments, active requests
            for key in snapshot_keys:
                self.snapshots_dict[req_num][key] = list()

    # TODO: Still need to save here
    def save_stats(self, file_type: str):
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
