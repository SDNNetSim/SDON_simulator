# Standard library imports
import os
import json

# Third party imports
import numpy as np
import networkx as nx

# Local application imports
from useful_functions.handle_dirs_files import create_dir
from useful_functions.sim_functions import find_path_congestion
from sim_scripts.routing import Routing


class QLearning:
    """
    Controls methods related to the Q-learning reinforcement learning algorithm.
    """

    def __init__(self, properties: dict):
        """
        Initializes the QLearning class.
        """
        self.properties = properties
        self.ai_arguments = properties['ai_arguments']
        if self.ai_arguments['is_training']:
            self.sim_type = 'train'
        else:
            self.sim_type = 'test'
        # Contains all state and action value pairs
        self.q_routes = None
        self.q_cores = None
        # Statistics to evaluate our reward function
        self.rewards_dict = {'average': [], 'min': [], 'max': [], 'rewards': {}}

        self.curr_episode = None
        self.num_nodes = None
        # Source node, destination node, and the resulting path
        self.source = None
        self.destination = None
        self.chosen_path = None
        # The chosen bandwidth for the current request
        self.chosen_bw = None
        # The latest up-to-date network spectrum database
        self.net_spec_db = None
        self.xt_worst = None
        self.k_paths = None
        self.num_cores = None
        self.q_routes = None
        self.path_index = None
        self.cong_index = None
        self.cong_types = None
        self.q_cores = None
        self.reward_policies = {
            'baseline': None,
        }
        # Simulation methods related to routing
        self.routing_obj = Routing(beta=properties['beta'], topology=properties['topology'],
                                   guard_slots=properties['guard_slots'])

    @staticmethod
    def set_seed(seed: int):
        np.random.seed(seed)

    def decay_epsilon(self, amount: float):
        self.ai_arguments['epsilon'] -= amount
        if self.ai_arguments['epsilon'] < 0.0:
            raise ValueError(f"Epsilon should be greater than 0 but it is {self.ai_arguments['epsilon']}")

    def _update_rewards_dict(self, reward: float = None):
        episode = str(self.curr_episode)
        if episode not in self.rewards_dict['rewards'].keys():
            self.rewards_dict['rewards'][episode] = [reward]
        else:
            self.rewards_dict['rewards'][episode].append(reward)

        if self.curr_episode == self.properties['max_iters'] - 1 and \
                len(self.rewards_dict['rewards'][episode]) == self.properties['num_requests']:
            matrix = np.array([])
            for episode, curr_list in self.rewards_dict['rewards'].items():
                if episode == '0':
                    matrix = np.array([curr_list])
                else:
                    matrix = np.vstack((matrix, curr_list))

            self.rewards_dict['min'] = matrix.min(axis=0, initial=np.inf).tolist()
            self.rewards_dict['max'] = matrix.max(axis=0, initial=np.inf * -1.0).tolist()
            self.rewards_dict['average'] = matrix.mean(axis=0).tolist()
            self.rewards_dict.pop('rewards')

    def save_table(self):
        if self.sim_type == 'test':
            raise NotImplementedError

        if self.ai_arguments['table_path'] == 'None':
            date_time = f"{self.properties['network']}/{self.properties['date']}/{self.properties['sim_start']}"
            self.ai_arguments['table_path'] = f"{date_time}/{self.properties['thread_num']}"

        create_dir(f"ai/models/q_tables/{self.ai_arguments['table_path']}")
        file_path = f"{os.getcwd()}/ai/models/q_tables/{self.ai_arguments['table_path']}/"
        file_name = f"{self.properties['erlang']}_table_c{self.properties['cores_per_link']}.npy"
        np.save(file_path + file_name, self.q_routes)

        properties_dict = {
            'epsilon': self.ai_arguments['epsilon'],
            'episodes': self.properties['max_iters'],
            'learn_rate': self.ai_arguments['learn_rate'],
            'discount_factor': self.ai_arguments['discount'],
            'reward_info': self.rewards_dict
        }
        file_name = f"{self.properties['erlang']}_params.json"
        with open(f"{file_path}/{file_name}", 'w', encoding='utf-8') as file:
            json.dump(properties_dict, file)

    def load_table(self):
        file_path = f"{os.getcwd()}/ai/models/q_tables/{self.ai_arguments['table_path']}/"
        file_name = f"{self.sim_type}_table_c{self.ai_arguments['cores_per_link']}.npy"
        try:
            self.q_routes = np.load(file_path + file_name)
        except FileNotFoundError:
            print('File not found, please ensure if you are testing, an already trained file exists and has been '
                  'specified correctly.')

        file_name = f"hyper_properties_c{self.ai_arguments['cores_per_link']}.json"
        with open(f"{file_path}{file_name}", encoding='utf-8') as file:
            properties_obj = json.load(file)

        self.ai_arguments['epsilon'] = properties_obj['epsilon']
        self.ai_arguments['learn_rate'] = properties_obj['learn_rate']
        self.ai_arguments['discount'] = properties_obj['discount_factor']
        self.rewards_dict = properties_obj['reward_info']

    def update_q_values(self, routed: bool, spectrum: dict, path_mod: str):
        self.update_routes_q_values(routed, path_mod)
        self.update_cores_q_values(spectrum, routed, path_mod)

    def update_routes_q_values(self, routed: bool, path_mod: str):
        policy = self.ai_arguments.get('policy')
        if policy not in self.reward_policies:
            raise NotImplementedError('Reward policy not recognized.')

        reward = self.reward_policies[policy](routed=routed, path_mod=path_mod)
        self._update_rewards_dict(reward=reward)

        source = int(self.chosen_path[0])
        destination = int(self.chosen_path[-1])
        path_index = np.where(self.q_routes['path'] == self.chosen_path)[0][0]

        current_q = self.q_routes[source][destination][path_index]['q_value']
        max_future_q_routes = np.max(self.q_routes[source][destination]['q_value'])

        new_q_routes = ((1.0 - self.ai_arguments['learn_rate']) * current_q) + \
                       (self.ai_arguments['learn_rate'] * (
                               reward + (self.ai_arguments['discount'] * max_future_q_routes)))

        self.q_routes[source][destination][path_index]['q_value'] = new_q_routes

    def update_cores_q_values(self, spectrum: dict, routed: bool, path_mod: str):
        policy = self.ai_arguments.get('policy')
        if policy not in self.reward_policies:
            raise NotImplementedError('Reward policy not recognized.')

        reward = self.reward_policies[policy](routed=routed, path_mod=path_mod)
        self._update_rewards_dict(reward=reward)

        source = int(self.chosen_path[0])
        destination = int(self.chosen_path[-1])
        path_index = np.where(self.q_routes['path'] == self.chosen_path)[0][0]
        core_index = np.where(self.q_cores['core_action'] == spectrum['core'])[0][0]

        current_q_core = self.q_cores[source][destination][path_index][core_index]['q_value']
        max_future_q_core = np.max(self.q_cores[source][destination][path_index]['q_value'])

        new_q_core = ((1.0 - self.ai_arguments['learn_rate']) * current_q_core) + \
                     (self.ai_arguments['learn_rate'] * (reward + (self.ai_arguments['discount'] * max_future_q_core)))

        self.q_cores[source][destination][path_index][core_index]['q_value'] = new_q_core

    def _init_q_tables(self):
        for source in range(0, self.num_nodes):
            for destination in range(0, self.num_nodes):
                # A node cannot be attached to itself
                if source == destination:
                    continue

                shortest_paths = list(nx.shortest_simple_paths(self.properties['topology'],
                                                               str(source), str(destination)))
                for k, curr_path in enumerate(shortest_paths):
                    if k >= self.k_paths:
                        break

                    for c_index, congestion in enumerate(self.cong_types):
                        self.q_routes[source, destination, k, c_index] = (curr_path, 0.0)

                        for core_action in range(self.num_cores):
                            self.q_cores[source, destination, k, c_index, core_action] = (curr_path, core_action, 0.0)

    def setup_env(self):
        self.num_nodes = len(list(self.properties['topology'].nodes()))
        self.k_paths = self.properties['k_paths']

        self.cong_types = ['Low', 'Medium', 'High']
        route_types = [('path', 'O'), ('q_value', 'f8')]
        core_types = [('path', 'O'), ('core_action', 'i8'), ('q_value', 'f8')]

        self.q_routes = np.empty((self.num_nodes, self.num_nodes, self.k_paths, len(self.cong_types)),
                                 dtype=route_types)
        self.num_cores = self.properties['cores_per_link']
        self.q_cores = np.empty((self.num_nodes, self.num_nodes, self.k_paths, len(self.cong_types), self.num_cores),
                                dtype=core_types)

        self._init_q_tables()

    def _get_max_q(self, paths):
        q_values = list()
        for path_index, _, cong_index in paths:
            curr_q = self.q_routes[self.source][self.destination][path_index][cong_index]['q_value']
            q_values.append(curr_q)

        max_index = np.argmax(q_values)
        max_path = paths[max_index]
        return max_path

    def _assign_congestion(self, paths):
        resp = list()
        for path_index, curr_path in enumerate(paths):
            curr_cong = find_path_congestion(path=curr_path, network_db=self.net_spec_db)
            if curr_cong < 0.3:
                cong_index = 0
            elif 0.3 <= curr_cong < 0.7:
                cong_index = 1
            elif curr_cong >= 0.7:
                cong_index = 2
            else:
                raise ValueError('Congestion value not recognized.')

            resp.append((path_index, curr_path, cong_index))

        return resp

    def route(self):
        random_float = np.round(np.random.uniform(0, 1), decimals=1)
        paths = self.q_routes[self.source][self.destination]['path'][:, 0]
        paths = self._assign_congestion(paths=paths)

        if random_float < self.ai_arguments['epsilon']:
            self.path_index = np.random.choice(self.k_paths)
            self.cong_index = np.random.choice(len(self.cong_types))
            self.chosen_path = self.q_routes[self.source][self.destination][self.path_index][self.cong_index]['path']
        else:
            best_path = self._get_max_q(paths=paths)
            self.path_index, self.chosen_path, self.cong_index = best_path

        if len(self.chosen_path) == 0:
            raise ValueError('The chosen path can not be None')

        return self.chosen_path

    def core_assignment(self):
        # TODO: You are not flagging for low, medium, and high (select based on those indexes only!)
        #   - Calculate congestion for each core :(
        random_float = np.round(np.random.uniform(0, 1), decimals=1)
        random_float = 111111

        if random_float < self.ai_arguments['epsilon']:
            chosen_core = np.random.randint(0, self.properties['cores_per_link'])
        else:
            q_values = self.q_cores[self.source][self.destination][self.path_index]['q_value']
            self.core_index = None
            chosen_core = self.q_cores[self.source][self.destination][self.path_index][np.argmax(q_values)][
                'core_action']

        return chosen_core
