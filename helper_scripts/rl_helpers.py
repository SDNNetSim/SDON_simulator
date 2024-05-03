import numpy as np

from helper_scripts.sim_helpers import find_path_len, get_path_mod, get_hfrag
from helper_scripts.sim_helpers import find_path_cong, classify_cong


# TODO: Might switch around the functions in this script to make more sense
#   - Something in a more organized way, separate QL from DQN, DQN from A2C, etc.
class RLHelpers:
    """
    Contains methods to assist with reinforcement learning simulations.
    """

    def __init__(self, rl_props: dict, engine_obj: object, route_obj: object, q_props: dict, drl_props: dict):
        # TODO: Check for variables used and unused
        #   - Improve naming
        self.rl_props = rl_props
        self.q_props = q_props
        self.drl_props = drl_props

        self.engine_obj = engine_obj
        self.route_obj = route_obj

        self.topology = None
        self.net_spec_dict = None
        self.reqs_status_dict = None
        self.algorithm = None
        self.completed_sim = False

        self.no_penalty = None
        self.path_index = None
        self.core_num = None
        self.slice_request = None
        self.super_channel = None
        self.super_channel_indexes = list()
        self.mod_format = None
        self.bandwidth = None

    def update_snapshots(self):
        arrival_count = self.rl_props['arrival_count']

        snapshot_step = self.engine_obj.engine_props['snapshot_step']
        if self.engine_obj.engine_props['save_snapshots'] and (arrival_count + 1) % snapshot_step == 0:
            self.engine_obj.stats_obj.update_snapshot(net_spec_dict=self.engine_obj.net_spec_dict,
                                                      req_num=arrival_count + 1)

    def get_super_channels(self, slots_needed: int, num_channels: int):
        """
        Gets the available 'J' super-channels for the agent to choose from along with a fragmentation score.

        :param slots_needed: Slots needed by the current request.
        :param num_channels: Number of channels needed by the current request.
        :return: A matrix of super-channels with their fragmentation score.
        :rtype: list
        """
        path_list = self.rl_props['paths_list'][self.rl_props['path_index']]
        sc_index_mat, hfrag_arr = get_hfrag(path_list=path_list, net_spec_dict=self.engine_obj.net_spec_dict,
                                            spectral_slots=self.rl_props['spectral_slots'], core_num=self.core_num,
                                            slots_needed=slots_needed)

        self.super_channel_indexes = sc_index_mat[0:num_channels]
        # There were not enough super-channels, do not penalize the agent
        if len(self.super_channel_indexes) < self.rl_props['super_channel_space']:
            self.no_penalty = True
        else:
            self.no_penalty = False

        resp_frag_mat = list()
        for channel in self.super_channel_indexes:
            start_index = channel[0]
            resp_frag_mat.append(hfrag_arr[start_index])

        resp_frag_mat = np.where(np.isinf(resp_frag_mat), 100.0, resp_frag_mat)
        difference = self.rl_props['super_channel_space'] - len(resp_frag_mat)

        if len(resp_frag_mat) < self.rl_props['super_channel_space'] or np.any(np.isinf(resp_frag_mat)):
            for _ in range(difference):
                resp_frag_mat = np.append(resp_frag_mat, 100.0)

        return resp_frag_mat

    def classify_paths(self, paths_list: list):
        info_list = list()
        paths_list = paths_list[:, 0]
        for path_index, curr_path in enumerate(paths_list):
            curr_cong = find_path_cong(path_list=curr_path, net_spec_dict=self.engine_obj.net_spec_dict)
            cong_index = classify_cong(curr_cong=curr_cong)

            info_list.append((path_index, curr_path, cong_index))

        return info_list

    def update_route_props(self, bandwidth: str, chosen_path: list):
        """
        Updates the route properties.

        :param bandwidth: Bandwidth of the current request.
        :param chosen_path: Path of the current request.
        """
        self.route_obj.route_props['paths_list'] = [chosen_path]
        path_len = find_path_len(path_list=chosen_path, topology=self.engine_obj.engine_props['topology'])
        mod_format = get_path_mod(mods_dict=self.engine_obj.engine_props['mod_per_bw'][bandwidth], path_len=path_len)
        self.route_obj.route_props['mod_formats_list'].append([mod_format])
        self.route_obj.route_props['weights_list'].append(path_len)

    def find_maximums(self):
        """
        Finds the maximum length a modulation can support and the number of slots.
        """
        for bandwidth, mod_obj in self.engine_obj.engine_props['mod_per_bw'].items():
            bandwidth_percent = self.engine_obj.engine_props['request_distribution'][bandwidth]
            if bandwidth_percent > 0:
                self.rl_props['bandwidth_list'].append(bandwidth)
            for _, data_obj in mod_obj.items():
                if data_obj['slots_needed'] > self.drl_props['max_slots_needed'] and bandwidth_percent > 0:
                    self.drl_props['max_slots_needed'] = data_obj['slots_needed']
                if data_obj['max_length'] > self.drl_props['max_length'] and bandwidth_percent > 0:
                    self.drl_props['max_length'] = data_obj['max_length']

    def handle_releases(self):
        """
        Checks if a request or multiple requests need to be released.
        """
        curr_time = self.rl_props['arrival_list'][self.rl_props['arrival_count']]['arrive']

        for _, req_obj in enumerate(self.rl_props['depart_list']):
            if req_obj['depart'] <= curr_time:
                self.engine_obj.handle_release(curr_time=req_obj['depart'])

    def allocate(self, route_obj: object):
        """
        Attempts to allocate a given request.

        :param route_obj: The Routing class.
        """
        curr_time = self.rl_props['arrival_list'][self.rl_props['arrival_count']]['arrive']
        # TODO: Check this for when using core agent and when not using
        self.engine_obj.handle_arrival(curr_time=curr_time, force_route_matrix=self.rl_props['chosen_path'],
                                       force_core=self.rl_props['core_index'])

    def update_mock_sdn(self, curr_req: dict):
        """
        Updates the mock sdn dictionary to find select routes.

        :param curr_req: The current request.
        :return: The mock return of the SDN controller.
        :rtype: dict
        """
        mock_sdn = {
            'req_id': curr_req['req_id'],
            'source': curr_req['source'],
            'destination': curr_req['destination'],
            'bandwidth': curr_req['bandwidth'],
            'net_spec_dict': self.net_spec_dict,
            'topology': self.topology,
            'mod_formats': curr_req['mod_formats'],
            'num_trans': 1.0,
            'route_time': 0.0,
            'block_reason': None,
            'stat_key_list': ['modulation_list', 'xt_list', 'core_list'],
            'modulation_list': list(),
            'xt_list': list(),
            'is_sliced': False,
            'core_list': list(),
            'bandwidth_list': list(),
            'path_weight': list(),
            'spectrum_dict': {'modulation': None}
        }

        return mock_sdn

    def reset_reqs_dict(self, seed: int):
        self.engine_obj.generate_requests(seed=seed)
        self.min_arrival = np.inf
        self.max_arrival = -1 * np.inf
        self.min_depart = np.inf
        self.max_depart = -1 * np.inf

        for req_time in self.engine_obj.reqs_dict:
            if self.engine_obj.reqs_dict[req_time]['request_type'] == 'arrival':
                if req_time > self.max_arrival:
                    self.max_arrival = req_time
                if req_time < self.min_arrival:
                    self.min_arrival = req_time

                self.rl_props['arrival_list'].append(self.engine_obj.reqs_dict[req_time])
            else:
                if req_time > self.max_depart:
                    self.max_depart = req_time
                if req_time < self.min_depart:
                    self.min_depart = req_time

                self.rl_props['depart_list'].append(self.engine_obj.reqs_dict[req_time])

    # TODO: This will probably also move to rl helpers
    def _error_check_actions(self):
        if self.helper_obj.path_index < 0 or self.helper_obj.path_index > (self.rl_props['k_paths'] - 1):
            raise ValueError(f'Path index out of range: {self.helper_obj.path_index}')
        if self.helper_obj.core_num < 0 or self.helper_obj.core_num > (
                self.rl_props['cores_per_link'] - 1):
            raise ValueError(f'Core index out of range: {self.helper_obj.core_num}')
