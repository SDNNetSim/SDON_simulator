import networkx as nx
import numpy as np
import os
import json

from arg_scripts.rl_args import empty_q_props
from helper_scripts.sim_helpers import find_path_cong, classify_cong, calc_matrix_stats, find_core_cong
from helper_scripts.os_helpers import create_dir


# TODO: Generalize as many functions as you can, probably will move QLearning to another script
# TODO: Probably need to standardize function names for this to be generalized
class QLearningHelpers:
    def __init__(self, rl_props: dict, engine_props: dict):
        # TODO: Props in its own object called props (standards and guidelines)
        self.props = empty_q_props
        self.engine_props = engine_props
        self.rl_props = rl_props

        self.path_levels = 3
        self.completed_sim = False
        self.iteration = 0

    def _init_q_tables(self):
        for source in range(0, self.rl_props['num_nodes']):
            for destination in range(0, self.rl_props['num_nodes']):
                # A node cannot be attached to itself
                if source == destination:
                    continue

                shortest_paths = nx.shortest_simple_paths(G=self.engine_props['topology'],
                                                          source=str(source), target=str(destination), weight='length')
                for k, curr_path in enumerate(shortest_paths):
                    if k >= self.rl_props['k_paths']:
                        break

                    for level_index in range(self.path_levels):
                        self.props['routes_matrix'][source, destination, k, level_index] = (curr_path, 0.0)

                        for core_action in range(self.engine_props['cores_per_link']):
                            core_tuple = (curr_path, core_action, 0.0)
                            self.props['cores_matrix'][k, core_action, level_index] = core_tuple

    def setup_env(self):
        self.props['epsilon'] = self.engine_props['epsilon_start']
        route_types = [('path', 'O'), ('q_value', 'f8')]
        core_types = [('path', 'O'), ('core_action', 'i8'), ('q_value', 'f8')]

        self.props['routes_matrix'] = np.empty((self.rl_props['num_nodes'], self.rl_props['num_nodes'],
                                                self.rl_props['k_paths'], self.path_levels), dtype=route_types)

        self.props['cores_matrix'] = np.empty((self.rl_props['k_paths'],
                                               self.engine_props['cores_per_link'], self.path_levels), dtype=core_types)

        self._init_q_tables()

    def get_max_future_q(self, path_list: list, net_spec_dict: dict, matrix: list, core: list, flag: str):
        if flag == 'path':
            new_cong = find_path_cong(path_list=path_list, net_spec_dict=net_spec_dict)
        elif flag == 'core':
            new_cong = find_core_cong(core=core)
        else:
            raise NotImplementedError

        new_cong_index = classify_cong(curr_cong=new_cong)
        max_future_q = matrix[self.rl_props['path_index']][new_cong_index]['q_value']
        return max_future_q

    def update_routes_matrix(self, reward: float, level_index: int, net_spec_dict: dict):
        routes_matrix = self.props['routes_matrix'][self.rl_props['source']][self.rl_props['destination']]
        path_list = routes_matrix[self.rl_props['path_index']][level_index]
        current_q = path_list['q_value']

        # TODO: Should NOT pass [0][0]
        max_future_q = self.get_max_future_q(path_list=routes_matrix[self.rl_props['path_index']][0][0],
                                             net_spec_dict=net_spec_dict, routes_matrix=routes_matrix)

        delta = reward + self.engine_props['discount_factor'] * max_future_q
        td_error = current_q - (reward + self.engine_props['discount_factor'] * max_future_q)
        self.update_q_stats(reward=reward, stats_flag='routes_dict', td_error=td_error)
        new_q = ((1.0 - self.engine_props['learn_rate']) * current_q) + (self.engine_props['learn_rate'] * delta)

        routes_matrix = self.props['routes_matrix'][self.rl_props['source']][self.rl_props['destination']]
        routes_matrix[self.rl_props['path_index']]['q_value'] = new_q

    def update_cores_matrix(self, reward: float, core_index: int, level_index: int, net_spec_dict: dict):
        cores_matrix = self.props['cores_matrix'][self.rl_props['source']][self.rl_props['destination']]
        path_list = cores_matrix[self.rl_props['path_index']][level_index][core_index]
        current_q = path_list['q_value']

        # TODO: Should NOT pass [0][0]
        max_future_q = self.get_max_future_q(path_list=cores_matrix[self.rl_props['path_index']][0][0],
                                             net_spec_dict=net_spec_dict, routes_matrix=cores_matrix)

        delta = reward + self.engine_props['discount_factor'] * max_future_q
        td_error = current_q - (reward + self.engine_props['discount_factor'] * max_future_q)
        self.update_q_stats(reward=reward, stats_flag='cores_dict', td_error=td_error)
        new_q = ((1.0 - self.engine_props['learn_rate']) * current_q) + (self.engine_props['learn_rate'] * delta)

        cores_matrix = self.props['cores_matrix'][self.rl_props['source']][self.rl_props['destination']]
        cores_matrix[self.rl_props['path_index']][core_index]['q_value'] = new_q

    def get_max_curr_q(self, paths_info):
        q_values = list()
        for path_index, _, level_index in paths_info:
            routes_matrix = self.props['routes_matrix'][self.rl_props['source']][self.rl_props['destination']]
            curr_q = routes_matrix[path_index][level_index]['q_value']
            q_values.append(curr_q)

        max_index = np.argmax(q_values)
        max_path = self.rl_props['paths_list'][max_index]
        return max_index, max_path

    def _calc_q_averages(self, stats_flag: str, episode: str):
        len_rewards = len(self.props['rewards_dict'][stats_flag]['rewards'][episode])

        max_iters = self.engine_props['max_iters']
        num_requests = self.engine_props['num_requests']

        if self.iteration == (max_iters - 1) and len_rewards == num_requests:
            self.completed_sim = True
            rewards_dict = self.props['rewards_dict'][stats_flag]['rewards']
            errors_dict = self.props['errors_dict'][stats_flag]['errors']
            self.props['rewards_dict'][stats_flag] = calc_matrix_stats(input_dict=rewards_dict)
            self.props['errors_dict'][stats_flag] = calc_matrix_stats(input_dict=errors_dict)

            self.save_model()

    def update_q_stats(self, reward: float, td_error: float, stats_flag: str):
        # To account for a reset even after a sim has completed (how SB3 works)
        if self.completed_sim:
            return

        episode = str(self.iteration)
        if episode not in self.props['rewards_dict'][stats_flag]['rewards'].keys():
            self.props['rewards_dict'][stats_flag]['rewards'][episode] = [reward]
            self.props['errors_dict'][stats_flag]['errors'][episode] = [td_error]
            self.props['sum_rewards_dict'][episode] = reward
            self.props['sum_errors_dict'][episode] = td_error
        else:
            self.props['rewards_dict'][stats_flag]['rewards'][episode].append(reward)
            self.props['errors_dict'][stats_flag]['errors'][episode].append(td_error)
            self.props['sum_rewards_dict'][episode] += reward
            self.props['sum_errors_dict'][episode] += td_error

        self._calc_q_averages(stats_flag=stats_flag, episode=episode)

    def _save_params(self, save_dir: str):
        params_dict = dict()
        for param_type, params_list in self.props['save_params_dict'].items():
            for curr_param in params_list:
                if param_type == 'engine_params_list':
                    params_dict[curr_param] = self.engine_props[curr_param]
                else:
                    params_dict[curr_param] = self.props[curr_param]

        erlang = self.engine_props['erlang']
        cores_per_link = self.engine_props['cores_per_link']
        param_fp = f"e{erlang}_params_c{cores_per_link}.json"
        param_fp = os.path.join(save_dir, param_fp)
        with open(param_fp, 'w', encoding='utf-8') as file_obj:
            json.dump(params_dict, file_obj)

    # TODO: Save every 'x' iters
    def save_model(self):
        """
        Saves the current q-learning model.
        """
        date_time = os.path.join(self.engine_props['network'], self.engine_props['date'],
                                 self.engine_props['sim_start'])
        save_dir = os.path.join('logs', 'ql', date_time)
        create_dir(file_path=save_dir)

        erlang = self.engine_props['erlang']
        cores_per_link = self.engine_props['cores_per_link']
        route_fp = f"e{erlang}_routes_c{cores_per_link}.npy"
        core_fp = f"e{erlang}_cores_c{cores_per_link}.npy"

        for curr_fp in (route_fp, core_fp):
            save_fp = os.path.join(os.getcwd(), save_dir, curr_fp)

            if curr_fp.split('_')[1] == 'routes':
                np.save(save_fp, self.props['routes_matrix'])
            else:
                np.save(save_fp, self.props['cores_matrix'])

        self._save_params(save_dir=save_dir)

    def decay_epsilon(self):
        if self.props['epsilon'] > self.engine_props['epsilon_end']:
            decay_rate = (self.engine_props['epsilon_start'] - self.engine_props['epsilon_end'])
            decay_rate /= self.engine_props['max_iters']
            self.props['epsilon'] -= decay_rate

        if self.iteration == 0:
            self.props['epsilon_list'].append(self.props['epsilon'])

        if self.props['epsilon'] < 0.0:
            raise ValueError(f"Epsilon should be greater than 0 but it is {self.props['epsilon']}")
