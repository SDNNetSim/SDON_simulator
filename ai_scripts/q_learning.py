# TODO: Add SIGINT and SIGTERM to ai_scripts functions
# TODO: Every function needs to be checked (debug prior then debug this one together)
# TODO: Calculate TD averages (matrix average in another script?)
# TODO: Calculate reward averages (matrix average in another script?)
# TODO: Update stats?
# TODO: "Finish this script", then move to ai_args, then move to engine, then config file
# TODO: Goal is to know if there are bugs by today, initial results tomorrow
# TODO: Get this working, then clean it up
# TODO: Test and update to team standards
import os
import json

import numpy as np
import networkx as nx

from helper_scripts.sim_helpers import find_path_len, get_path_mod
from helper_scripts.os_helpers import create_dir
from arg_scripts.ai_args import empty_q_props


class QLearning:

    def __init__(self, engine_props: dict):
        # TODO: Props needs to be initialized
        self.q_props = empty_q_props
        self.engine_props = engine_props

        # TODO: Better way to do this? I think access in engine props instead
        self.num_nodes = None
        self.k_paths = None

        # TODO: Something else like sdn props?
        self.source = None
        self.destination = None

        self.chosen_path = None
        self.chosen_bw = None
        self.core_index = None
        # TODO: What is this variable?
        self.paths_info = None

        # TODO: Double check these
        self.routing_obj = None
        self.curr_episode = None
        self.paths_list = None
        self.path_index = None

    @staticmethod
    def set_seed(seed: int):
        np.random.seed(seed)

    def decay_epsilon(self, amount: float):
        self.q_props['epsilon'] -= amount
        if self.curr_episode == 0:
            self.q_props['epsilon_list'].append(self.q_props['epsilon'])

        if self.q_props['epsilon'] < 0.0:
            raise ValueError(f"Epsilon should be greater than 0 but it is {self.q_props['epsilon']}")

    def _save_params(self, save_dir: str):
        params_dict = dict()
        for param_type, params_list in self.q_props['save_params_dict'].items():
            for curr_param in params_list:
                if param_type == 'engine_params_list':
                    params_dict[curr_param] = self.engine_props[curr_param]
                else:
                    params_dict[curr_param] = self.q_props[curr_param]

        param_fp = f"e{self.engine_props['erlang']}_params_c{self.engine_props['cores_per_link']}.json"
        param_fp = os.path.join(save_dir, param_fp)
        with open(param_fp, 'w', encoding='utf-8') as file_obj:
            json.dump(params_dict, file_obj)

    def save_model(self):
        date_time = os.path.join(self.engine_props['network'], self.engine_props['date'],
                                 self.engine_props['sim_start'])
        save_dir = os.path.join('data', 'ai', 'models', date_time)
        create_dir(file_path=save_dir)

        route_fp = f"e{self.engine_props['erlang']}_routes_c{self.engine_props['cores_per_link']}.npy"
        core_fp = f"e{self.engine_props['erlang']}_cores_c{self.engine_props['cores_per_link']}.npy"

        for curr_fp in (route_fp, core_fp):
            save_fp = os.path.join(os.getcwd(), save_dir, curr_fp)

            if curr_fp.split('_')[1] == 'routes':
                np.save(save_fp, 'routes_matrix')
            else:
                np.save(save_fp, self.q_props['cores_matrix'])

        self._save_params(save_dir=save_dir)

    def load_model(self):
        raise NotImplementedError

    def _update_routes_q_values(self, routed: bool):
        policy = self.ai_arguments.get('policy')
        if policy not in self.reward_policies:
            raise NotImplementedError('Reward policy not recognized.')

        new_path_cong = find_path_congestion(path=self.chosen_path, network_db=self.net_spec_db)
        current_q = self.q_routes[self.source][self.destination][self.path_index][self.cong_index]['q_value']
        max_future_q = self._get_max_future_q(new_cong=new_path_cong)

        reward = self.reward_policies[policy](routed=routed)
        delta = reward + self.ai_arguments['discount'] * max_future_q

        td_error = current_q - (reward + self.ai_arguments['discount'] * max_future_q)
        self._update_stats(reward=reward, reward_flag='routes', td_error=td_error)

        new_q = ((1.0 - self.ai_arguments['learn_rate']) * current_q) + (self.ai_arguments['learn_rate'] * delta)
        self.q_routes[self.source][self.destination][self.path_index][self.cong_index]['q_value'] = new_q

    def _update_cores_q_values(self, routed: bool):
        policy = self.ai_arguments.get('policy')
        if policy not in self.reward_policies:
            raise NotImplementedError('Reward policy not recognized.')

        q_cores_matrix = self.q_cores[self.source][self.destination][self.path_index]
        current_q = q_cores_matrix[self.cong_index][self.core_index]['q_value']
        max_future_q = np.max(q_cores_matrix[self.new_cong_index]['q_value'])

        reward = self.reward_policies[policy](routed=routed)
        delta = reward + self.ai_arguments['discount'] * max_future_q
        self._update_stats(reward=reward, reward_flag='cores', td_error=delta)

        new_q_core = ((1.0 - self.ai_arguments['learn_rate']) * current_q) + (self.ai_arguments['learn_rate'] * delta)
        self.q_cores[self.source][self.destination][self.path_index][self.cong_index][self.core_index][
            'q_value'] = new_q_core

    def update_env(self):
        self._update_routes()
        self._update_cores()

    def _init_q_tables(self):
        for source in range(0, self.num_nodes):
            for destination in range(0, self.num_nodes):
                # A node cannot be attached to itself
                if source == destination:
                    continue

                shortest_paths = nx.shortest_simple_paths(G=self.engine_props['topology'], source=str(source),
                                                          target=str(destination), weight='length')
                for k, curr_path in enumerate(shortest_paths):
                    if k >= self.k_paths:
                        break
                    self.q_props['routes_matrix'][source, destination, k] = (curr_path, 0.0)

                    for core_action in range(self.engine_props['cores_per_link']):
                        self.q_props['cores_matrix'][source, destination, k, core_action] = (curr_path, core_action,
                                                                                             0.0)

    def setup_env(self):
        self.q_props['epsilon'] = self.engine_props['epsilon_start']
        self.num_nodes = len(self.engine_props['topology_info']['nodes'].keys())
        self.k_paths = int(self.engine_props['k_paths'])

        route_types = [('path', 'O'), ('q_value', 'f8')]
        core_types = [('path', 'O'), ('core_action', 'i8'), ('q_value', 'f8')]

        self.q_props['routes_matrix'] = np.empty((self.num_nodes, self.num_nodes, self.k_paths), dtype=route_types)
        self.q_props['cores_matrix'] = np.empty((self.num_nodes, self.num_nodes, self.k_paths,
                                                 self.engine_props['cores_per_link']), dtype=core_types)

        self._init_q_tables()

    # TODO: Needs to change based on new formulation
    def _get_max_q(self):
        q_values = list()
        for path_index, _ in self.paths_list:
            curr_q = self.q_props['routes_matrix'][self.source][self.destination][path_index]['q_value']
            q_values.append(curr_q)

        max_index = np.argmax(q_values)
        max_path = self.paths_list[max_index]
        return max_index, max_path

    def _update_route_props(self, sdn_props: dict, route_props: dict):
        route_props['paths_list'].append(self.chosen_path)
        path_len = find_path_len(path_list=self.chosen_path, topology=self.engine_props['topology'])
        chosen_bw = sdn_props['bandwidth']
        mod_format = get_path_mod(mods_dict=self.engine_props['mod_per_bw'][chosen_bw], path_len=path_len)
        route_props['mod_formats_list'].append([mod_format])
        route_props['weights_list'].append(path_len)

    # TODO: Don't think we need route props
    # TODO: Return route and modulation format in needed format
    # TODO: Modify route props
    def get_route(self, sdn_props: dict, route_props: dict):
        random_float = np.round(np.random.uniform(0, 1), decimals=1)
        # TODO: How is source and destination updated?

        # TODO: Not sure if I want to use these like this (source, destination)
        self.source = int(sdn_props['source'])
        self.destination = int(sdn_props['destination'])
        self.paths_list = self.q_props['routes_matrix'][self.source][self.destination]['path']

        if self.paths_list.ndim != 1:
            self.paths_list = self.paths_list[:, 0]

        if random_float < self.q_props['epsilon']:
            self.path_index = np.random.choice(self.k_paths)
            self.chosen_path = self.paths_list[self.path_index]
        else:
            # TODO: Remember get max q will change based on my proposal
            best_path = self._get_max_q()
            self.path_index, self.chosen_path = best_path

        if len(self.chosen_path) == 0:
            raise ValueError('The chosen path can not be None')

        self._update_route_props(sdn_props=sdn_props, route_props=route_props)

    # TODO: Also modify max future Q here
    def get_core(self, spectrum_props: dict):
        random_float = np.round(np.random.uniform(0, 1), decimals=1)

        if random_float < self.q_props['epsilon']:
            self.core_index = np.random.randint(0, self.engine_props['cores_per_link'])
        else:
            # TODO: Remember max future q will come from something different
            q_values = self.q_props['cores_matrix'][self.source][self.destination]['q_value']
            self.core_index = np.argmax(q_values)

        spectrum_props['forced_core'] = self.core_index
