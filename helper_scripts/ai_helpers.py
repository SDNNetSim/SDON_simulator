import os
import json

import numpy as np
from gymnasium import spaces

from helper_scripts.os_helpers import create_dir
from helper_scripts.sim_helpers import find_path_len, find_core_frag_cong, calc_matrix_stats


class AIHelpers:
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

        self.path_index = None
        self.core_num = None
        self.slice_request = None
        self.mod_format = None
        self.bandwidth = None

    def _calc_q_averages(self, stats_flag: str, episode: str, iteration: int):
        len_rewards = len(self.q_props['rewards_dict'][stats_flag]['rewards'][episode])

        max_iters = self.engine_obj.engine_props['max_iters'] - 1
        num_requests = self.engine_obj.engine_props['num_requests']
        if iteration == max_iters and len_rewards == num_requests:
            rewards_dict = self.q_props['rewards_dict'][stats_flag]['rewards']
            errors_dict = self.q_props['errors_dict'][stats_flag]['errors']
            self.q_props['rewards_dict'][stats_flag] = calc_matrix_stats(input_dict=rewards_dict)
            self.q_props['errors_dict'][stats_flag] = calc_matrix_stats(input_dict=errors_dict)

    def update_q_stats(self, reward: float, td_error: float, stats_flag: str, iteration: int):
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

        # TODO: This should be accessible in props
        erlang = self.engine_obj.engine_props['erlang']
        cores_per_link = self.engine_obj.engine_props['cores_per_link']
        param_fp = f"e{erlang}_params_c{cores_per_link}.json"
        param_fp = os.path.join(save_dir, param_fp)
        with open(param_fp, 'w', encoding='utf-8') as file_obj:
            json.dump(params_dict, file_obj)

    # TODO: Change save directory
    # TODO: When to call save model
    def save_model(self):
        date_time = os.path.join(self.engine_obj.engine_props['network'], self.engine_obj.engine_props['date'],
                                 self.engine_obj.engine_props['sim_start'])
        save_dir = os.path.join('data', 'ai', 'models', date_time)
        create_dir(file_path=save_dir)

        # TODO: Again these should be accessible in another props, it's done twice
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

    def get_spectrum(self, paths_matrix: list):
        # To add core and fragmentation scores, make a k_path by cores by two matrix (two metrics)
        spectrum_matrix = np.zeros(
            (self.ai_props['drl_props']['k_paths'], self.ai_props['drl_props']['cores_per_link'], 2))
        for path_index, paths_list in enumerate(paths_matrix):
            for link_tuple in zip(paths_list, paths_list[1:]):
                rev_link_tuple = link_tuple[1], link_tuple[0]
                link_dict = self.engine_obj.net_spec_dict[link_tuple]
                rev_link_dict = self.engine_obj.net_spec_dict[rev_link_tuple]

                if link_dict != rev_link_dict:
                    raise ValueError('Link is not bi-directionally equal.')

                for core_index, core_arr in enumerate(link_dict['cores_matrix']):
                    spectrum_matrix[path_index][core_index] = find_core_frag_cong(
                        net_spec_db=self.engine_obj.net_spec_dict, path=paths_list, core=core_index)

        return spectrum_matrix

    def _calc_deep_reward(self, was_allocated: bool):
        if was_allocated:
            if self.slice_request:
                req_dict = self.ai_props['drl_props']['arrival_list'][self.ai_props['drl_props']['arrival_count']]
                max_reach = req_dict['mod_formats']['QPSK']['max_length']
                path_list = self.route_obj.route_props['paths_list'][self.path_index]
                path_len = find_path_len(topology=self.engine_obj.topology, path_list=path_list)

                # Did not have to slice
                if max_reach > path_len:
                    reward = 0.5
                else:
                    reward = 1.0
            else:
                reward = 1.0
        else:
            # Could have sliced and we did not
            if not self.slice_request:
                reward = -5.0
            else:
                reward = -1.0

        return reward

    def calculate_reward(self, was_allocated: bool):
        drl_reward = self._calc_deep_reward(was_allocated=was_allocated)
        ql_reward = None

        return drl_reward, ql_reward

    def find_maximums(self):
        for bandwidth, mod_obj in self.engine_obj.engine_props['mod_per_bw'].items():
            bandwidth_percent = self.engine_obj.engine_props['request_distribution'][bandwidth]
            if bandwidth_percent > 0:
                self.ai_props['drl_props']['bandwidth_list'].append(bandwidth)
            for modulation, data_obj in mod_obj.items():
                if data_obj['slots_needed'] > self.ai_props['drl_props']['max_slots_needed'] and bandwidth_percent > 0:
                    self.ai_props['drl_props']['max_slots_needed'] = data_obj['slots_needed']
                if data_obj['max_length'] > self.ai_props['drl_props']['max_length'] and bandwidth_percent > 0:
                    self.ai_props['drl_props']['max_length'] = data_obj['max_length']

    def get_obs_space(self):
        resp_obs = spaces.Dict({
            'source': spaces.Discrete(self.ai_props['drl_props']['num_nodes'], start=0),
            'destination': spaces.Discrete(self.ai_props['drl_props']['num_nodes'], start=0),
            'bandwidth': spaces.MultiBinary(len(self.ai_props['drl_props']['bandwidth_list'])),
            'cores_matrix': spaces.Box(low=0.01, high=1.01, shape=(self.ai_props['drl_props']['k_paths'],
                                                                   self.ai_props['drl_props']['cores_per_link'], 2)),
        })

        return resp_obs

    def get_action_space(self):
        action_space = spaces.MultiDiscrete([self.ai_props['drl_props']['k_paths'],
                                             self.ai_props['drl_props']['cores_per_link'],
                                             self.ai_props['drl_props']['slice_space']])

        return action_space

    def handle_releases(self):
        """
        Checks if a request or multiple requests need to be released.
        """
        curr_time = self.ai_props['drl_props']['arrival_list'][self.ai_props['drl_props']['arrival_count']]['arrive']
        index_list = list()

        for i, req_obj in enumerate(self.ai_props['drl_props']['depart_list']):
            if req_obj['depart'] <= curr_time:
                index_list.append(i)
                self.engine_obj.handle_release(curr_time=req_obj['depart'])

        for index in index_list:
            self.ai_props['drl_props']['depart_list'].pop(index)

    def allocate(self, route_obj: object):
        """
        Attempts to allocate a given request.

        :param route_obj: The Routing class.
        """
        path_matrix = [route_obj.route_props['paths_list'][self.path_index]]
        curr_time = self.ai_props['drl_props']['arrival_list'][self.ai_props['drl_props']['arrival_count']]['arrive']
        self.engine_obj.handle_arrival(curr_time=curr_time, force_route_matrix=path_matrix,
                                       force_slicing=self.slice_request)

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
