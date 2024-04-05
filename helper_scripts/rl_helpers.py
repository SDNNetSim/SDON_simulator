import os
import json

import numpy as np
from gymnasium import spaces

from helper_scripts.os_helpers import create_dir
from helper_scripts.sim_helpers import find_path_len, combine_and_one_hot, calc_matrix_stats, get_path_mod, get_hfrag


class RLHelpers:
    """
    Contains methods to assist with AI simulations.
    """

    def __init__(self, ai_props: dict, engine_obj: object, route_obj: object, q_props: dict, drl_props: dict):
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

        self.path_index = None
        self.core_num = None
        self.slice_request = None
        self.super_channel = None
        self.super_channel_indexes = list()
        self.mod_format = None
        self.bandwidth = None

    def get_super_channels(self, slots_needed: int, num_channels: int):
        path_list = self.ai_props['paths_list'][self.ai_props['path_index']]
        sc_index_mat, hfrag_arr = get_hfrag(path_list=path_list, net_spec_dict=self.engine_obj.net_spec_dict,
                                            spectral_slots=self.ai_props['spectral_slots'], core_num=self.core_num,
                                            slots_needed=slots_needed)

        self.super_channel_indexes = sc_index_mat[0:num_channels]
        resp_frag_mat = list()
        for channel in self.super_channel_indexes:
            start_index = channel[0]
            resp_frag_mat.append(hfrag_arr[start_index])

        return resp_frag_mat

    def get_max_curr_q(self):
        q_values = list()
        for path_index in range(len(self.ai_props['paths_list'])):
            routes_matrix = self.q_props['routes_matrix'][self.ai_props['source']][self.ai_props['destination']]
            curr_q = routes_matrix[path_index]['q_value']
            q_values.append(curr_q)

        max_index = np.argmax(q_values)
        max_path = self.ai_props['paths_list'][max_index]
        return max_index, max_path

    def update_route_props(self, bandwidth, chosen_path):
        # TODO: Check route props
        self.route_obj.route_props['paths_list'].append(chosen_path)
        path_len = find_path_len(path_list=chosen_path, topology=self.engine_obj.engine_props['topology'])
        chosen_bw = bandwidth
        mod_format = get_path_mod(mods_dict=self.engine_obj.engine_props['mod_per_bw'][chosen_bw], path_len=path_len)
        self.route_obj.route_props['mod_formats_list'].append([mod_format])
        self.route_obj.route_props['weights_list'].append(path_len)

    def _calc_q_averages(self, stats_flag: str, episode: str, iteration: int):
        len_rewards = len(self.q_props['rewards_dict'][stats_flag]['rewards'][episode])

        max_iters = self.engine_obj.engine_props['max_iters']
        num_requests = self.engine_obj.engine_props['num_requests']
        if iteration == max_iters and len_rewards == num_requests:
            self.completed_sim = True
            rewards_dict = self.q_props['rewards_dict'][stats_flag]['rewards']
            errors_dict = self.q_props['errors_dict'][stats_flag]['errors']
            self.q_props['rewards_dict'][stats_flag] = calc_matrix_stats(input_dict=rewards_dict)
            self.q_props['errors_dict'][stats_flag] = calc_matrix_stats(input_dict=errors_dict)

            self.save_model()

    def update_q_stats(self, reward: float, td_error: float, stats_flag: str, iteration: int):
        if self.completed_sim:
            return

        episode = str(iteration)
        if episode not in self.q_props['rewards_dict'][stats_flag]['rewards'].keys():
            self.q_props['rewards_dict'][stats_flag]['rewards'][episode] = [reward]
            self.q_props['errors_dict'][stats_flag]['errors'][episode] = [td_error]
            self.q_props['sum_rewards_dict'][episode] = reward
            self.q_props['sum_errors_dict'][episode] = td_error
        else:
            self.q_props['rewards_dict'][stats_flag]['rewards'][episode].append(reward)
            self.q_props['errors_dict'][stats_flag]['errors'][episode].append(td_error)
            self.q_props['sum_rewards_dict'][episode] += reward
            self.q_props['sum_errors_dict'][episode] += td_error

        self._calc_q_averages(stats_flag=stats_flag, episode=episode, iteration=iteration)

    def _save_params(self, save_dir: str):
        params_dict = dict()
        for param_type, params_list in self.q_props['save_params_dict'].items():
            for curr_param in params_list:
                if param_type == 'engine_params_list':
                    params_dict[curr_param] = self.engine_obj.engine_props[curr_param]
                else:
                    params_dict[curr_param] = self.q_props[curr_param]

        erlang = self.engine_obj.engine_props['erlang']
        cores_per_link = self.engine_obj.engine_props['cores_per_link']
        param_fp = f"e{erlang}_params_c{cores_per_link}.json"
        param_fp = os.path.join(save_dir, param_fp)
        with open(param_fp, 'w', encoding='utf-8') as file_obj:
            json.dump(params_dict, file_obj)

    # TODO: Save every x iters
    def save_model(self):
        date_time = os.path.join(self.engine_obj.engine_props['network'], self.engine_obj.engine_props['sim_start'])
        save_dir = os.path.join('logs', 'ql', date_time)
        create_dir(file_path=save_dir)

        erlang = self.engine_obj.engine_props['erlang']
        cores_per_link = self.engine_obj.engine_props['cores_per_link']
        route_fp = f"e{erlang}_routes_c{cores_per_link}.npy"
        core_fp = f"e{erlang}_cores_c{cores_per_link}.npy"

        for curr_fp in (route_fp, core_fp):
            save_fp = os.path.join(os.getcwd(), save_dir, curr_fp)

            if curr_fp.split('_')[1] == 'routes':
                np.save(save_fp, self.q_props['routes_matrix'])
            else:
                np.save(save_fp, self.q_props['cores_matrix'])

        self._save_params(save_dir=save_dir)

    def decay_epsilon(self, amount: float, iteration: int):
        self.q_props['epsilon'] -= amount
        if iteration == 0:
            self.q_props['epsilon_list'].append(self.q_props['epsilon'])

        if self.q_props['epsilon'] < 0.0:
            raise ValueError(f"Epsilon should be greater than 0 but it is {self.q_props['epsilon']}")

    def get_spectrum(self):
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
        drl_reward = self._calc_deep_reward(was_allocated=was_allocated)

        return drl_reward

    def find_maximums(self):
        for bandwidth, mod_obj in self.engine_obj.engine_props['mod_per_bw'].items():
            bandwidth_percent = self.engine_obj.engine_props['request_distribution'][bandwidth]
            if bandwidth_percent > 0:
                self.ai_props['bandwidth_list'].append(bandwidth)
            for modulation, data_obj in mod_obj.items():
                if data_obj['slots_needed'] > self.drl_props['max_slots_needed'] and bandwidth_percent > 0:
                    self.drl_props['max_slots_needed'] = data_obj['slots_needed']
                if data_obj['max_length'] > self.drl_props['max_length'] and bandwidth_percent > 0:
                    self.drl_props['max_length'] = data_obj['max_length']

    def get_obs_space(self, super_channel_space: int):
        self.find_maximums()
        resp_obs = spaces.Dict({
            'slots_needed': spaces.Discrete(self.drl_props['max_slots_needed'] + 1),
            'source': spaces.MultiBinary(self.ai_props['num_nodes']),
            'destination': spaces.MultiBinary(self.ai_props['num_nodes']),
            # TODO: Scale from 0-1, change observation space
            'super_channels': spaces.Discrete(super_channel_space)
        })

        return resp_obs

    @staticmethod
    def get_action_space(super_channel_space: int):
        super_channel_space = super_channel_space
        action_space = spaces.Discrete(super_channel_space)
        return action_space

    def handle_releases(self):
        """
        Checks if a request or multiple requests need to be released.
        """
        curr_time = self.ai_props['arrival_list'][self.ai_props['arrival_count']]['arrive']
        index_list = list()

        for i, req_obj in enumerate(self.ai_props['depart_list']):
            if req_obj['depart'] <= curr_time:
                index_list.append(i)
                self.engine_obj.handle_release(curr_time=req_obj['depart'])

        for index in index_list:
            self.ai_props['depart_list'].pop(index)

    def allocate(self, route_obj: object):
        """
        Attempts to allocate a given request.

        :param route_obj: The Routing class.
        """
        path_matrix = [route_obj.route_props['paths_list'][self.path_index]]
        curr_time = self.ai_props['arrival_list'][self.ai_props['arrival_count']]['arrive']
        self.engine_obj.handle_arrival(curr_time=curr_time, force_route_matrix=path_matrix,
                                       forced_index=self.start_index)

    def update_mock_sdn(self, curr_req: dict):
        """
        Updates the mock sdn dictionary to find select routes.
        :param curr_req: The current request.
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
