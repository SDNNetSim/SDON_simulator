import os
import json

import numpy as np
import networkx as nx
from gymnasium import spaces

from helper_scripts.os_helpers import create_dir
from helper_scripts.sim_helpers import find_path_len, combine_and_one_hot, calc_matrix_stats, get_path_mod, get_hfrag
from helper_scripts.sim_helpers import find_path_cong


# TODO: Might switch around the functions in this script to make more sense
#   - Something in a more organized way, separate QL from DQN, DQN from A2C, etc.
class RLHelpers:
    """
    Contains methods to assist with reinforcement learning simulations.
    """

    def __init__(self, ai_props: dict, engine_obj: object, route_obj: object, q_props: dict, drl_props: dict):
        # TODO: Check for variables used and unused
        #   - Improve naming
        self.ai_props = ai_props
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

    def _update_snapshots(self):
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
        path_list = self.ai_props['paths_list'][self.ai_props['path_index']]
        sc_index_mat, hfrag_arr = get_hfrag(path_list=path_list, net_spec_dict=self.engine_obj.net_spec_dict,
                                            spectral_slots=self.ai_props['spectral_slots'], core_num=self.core_num,
                                            slots_needed=slots_needed)

        self.super_channel_indexes = sc_index_mat[0:num_channels]
        # There were not enough super-channels, do not penalize the agent
        if len(self.super_channel_indexes) < self.ai_props['super_channel_space']:
            self.no_penalty = True
        else:
            self.no_penalty = False

        resp_frag_mat = list()
        for channel in self.super_channel_indexes:
            start_index = channel[0]
            resp_frag_mat.append(hfrag_arr[start_index])

        resp_frag_mat = np.where(np.isinf(resp_frag_mat), 100.0, resp_frag_mat)
        difference = self.ai_props['super_channel_space'] - len(resp_frag_mat)

        if len(resp_frag_mat) < self.ai_props['super_channel_space'] or np.any(np.isinf(resp_frag_mat)):
            for _ in range(difference):
                resp_frag_mat = np.append(resp_frag_mat, 100.0)

        return resp_frag_mat

    @staticmethod
    def _classify_cong(curr_cong):
        if curr_cong < 0.3:
            cong_index = 0
        elif 0.3 <= curr_cong < 0.7:
            cong_index = 1
        elif curr_cong >= 0.7:
            cong_index = 2
        else:
            raise ValueError('Congestion value not recognized.')

        return cong_index

    def classify_paths(self, paths_list: list):
        info_list = list()
        paths_list = paths_list[:, 0]
        for path_index, curr_path in enumerate(paths_list):
            curr_cong = find_path_cong(path_list=curr_path, net_spec_dict=self.engine_obj.net_spec_dict)
            cong_index = self._classify_cong(curr_cong=curr_cong)

            info_list.append((path_index, curr_path, cong_index))

        return info_list

    def update_route_props(self, bandwidth: str, chosen_path: list):
        """
        Updates the route properties.

        :param bandwidth: Bandwidth of the current request.
        :param chosen_path: Path of the current request.
        """
        self.route_obj.route_props['paths_list'].append(chosen_path)
        path_len = find_path_len(path_list=chosen_path, topology=self.engine_obj.engine_props['topology'])
        mod_format = get_path_mod(mods_dict=self.engine_obj.engine_props['mod_per_bw'][bandwidth], path_len=path_len)
        self.route_obj.route_props['mod_formats_list'].append([mod_format])
        self.route_obj.route_props['weights_list'].append(path_len)

    def get_spectrum(self):
        """
        Returns the spectrum as a binary array along a path.
        A one indicates that channel is taken along one or multiple of the links, a zero indicates that the channel
        is free along every link in the path.

        :return: The binary array of current path occupation.
        :rtype: list
        """
        resp_spec_arr = np.zeros(self.engine_obj.engine_props['spectral_slots'])
        path_list = self.ai_props['paths_list'][self.ai_props['path_index']]
        core_index = self.ai_props['core_index']
        net_spec_dict = self.engine_obj.net_spec_dict
        for source, dest in zip(path_list, path_list[1:]):
            core_arr = net_spec_dict[(source, dest)]['cores_matrix'][core_index]
            resp_spec_arr = combine_and_one_hot(resp_spec_arr, core_arr)

        return resp_spec_arr

    @staticmethod
    def _calc_deep_reward(was_allocated: bool):
        if was_allocated:
            reward = 1.0
        else:
            reward = -1.0

        return reward

    def calculate_drl_reward(self, was_allocated: bool):
        """
        Gets the reward for the deep reinforcement learning agent.

        :param was_allocated: Determines if the request was successfully allocated or not.
        :return: The reward.
        :rtype: float
        """
        if self.no_penalty:
            drl_reward = 0.0
        else:
            drl_reward = self._calc_deep_reward(was_allocated=was_allocated)

        return drl_reward

    def find_maximums(self):
        """
        Finds the maximum length a modulation can support and the number of slots.
        """
        for bandwidth, mod_obj in self.engine_obj.engine_props['mod_per_bw'].items():
            bandwidth_percent = self.engine_obj.engine_props['request_distribution'][bandwidth]
            if bandwidth_percent > 0:
                self.ai_props['bandwidth_list'].append(bandwidth)
            for _, data_obj in mod_obj.items():
                if data_obj['slots_needed'] > self.drl_props['max_slots_needed'] and bandwidth_percent > 0:
                    self.drl_props['max_slots_needed'] = data_obj['slots_needed']
                if data_obj['max_length'] > self.drl_props['max_length'] and bandwidth_percent > 0:
                    self.drl_props['max_length'] = data_obj['max_length']

    def get_obs_space(self):
        """
        Gets the observation space for the DRL agent.

        :return: The observation space.
        :rtype: spaces.Dict
        """
        self.find_maximums()
        resp_obs = spaces.Dict({
            'slots_needed': spaces.Discrete(self.drl_props['max_slots_needed'] + 1),
            'source': spaces.MultiBinary(self.ai_props['num_nodes']),
            'destination': spaces.MultiBinary(self.ai_props['num_nodes']),
            # TODO: Change
            'super_channels': spaces.Box(-0.01, 100.0, shape=(3,), dtype=np.float32)
        })

        return resp_obs

    # TODO: Change
    @staticmethod
    def get_action_space(super_channel_space: int = 3):
        """
        Gets the action space for the DRL agent.

        :param super_channel_space: The number of 'J' super-channels that can be selected.
        :return: The action space.
        :rtype: spaces.Discrete
        """
        action_space = spaces.Discrete(super_channel_space)
        return action_space

    def handle_releases(self):
        """
        Checks if a request or multiple requests need to be released.
        """
        curr_time = self.ai_props['arrival_list'][self.ai_props['arrival_count']]['arrive']

        for _, req_obj in enumerate(self.ai_props['depart_list']):
            if req_obj['depart'] <= curr_time:
                self.engine_obj.handle_release(curr_time=req_obj['depart'])

    def allocate(self, route_obj: object):
        """
        Attempts to allocate a given request.

        :param route_obj: The Routing class.
        """
        path_matrix = [route_obj.route_props['paths_list'][self.path_index]]
        curr_time = self.ai_props['arrival_list'][self.ai_props['arrival_count']]['arrive']

        # The spectrum was almost to maximum capacity, there will be blocking but it's not the agent's fault
        # Put the start index to zero (which will block regardless of what it is), but don't penalize the agent
        # if self.no_penalty:
        #     start_index = 0
        # else:
        #     start_index = self.super_channel_indexes[self.super_channel][0]

        # TODO: Got rid of forced index
        self.engine_obj.handle_arrival(curr_time=curr_time, force_route_matrix=path_matrix)

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
