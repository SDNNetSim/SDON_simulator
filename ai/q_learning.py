# Standard library imports
import os
import json

# Third party imports
import numpy as np
import networkx as nx

# Local application imports
from helper_scripts.os_helpers import create_dir
from helper_scripts.sim_helpers import find_path_congestion, find_core_frag_cong
from sim_scripts.routing import Routing


# TODO: Add SIGINT and SIGTERM to ai functions
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
        self.epsilon = self.ai_arguments['epsilon']
        if self.ai_arguments['is_training']:
            self.sim_type = 'train'
        else:
            self.sim_type = 'test'
        # Statistics to evaluate our reward function
        self.rewards_dict = {
            'routes': {'average': [], 'min': [], 'max': [], 'rewards': {}},
            'cores': {'average': [], 'min': [], 'max': [], 'rewards': {}}
        }
        self.td_errors = {
            'routes': {'average': [], 'min': [], 'max': [], 'errors': {}},
            'cores': {'average': [], 'min': [], 'max': [], 'errors': {}}
        }
        self.sum_rewards = dict()
        self.epsilon_stats = [self.epsilon]

        self.curr_episode = None
        self.num_nodes = None
        self.k_paths = None
        self.num_cores = None
        # The latest up-to-date network spectrum database
        self.net_spec_db = None

        self.q_routes = None
        self.q_cores = None

        # Source node, destination node, and the resulting path
        self.source = None
        self.destination = None
        # Contains all state and action value pairs
        self.chosen_path = None
        # The chosen bandwidth for the current request
        self.chosen_bw = None
        self.core_index = None
        self.paths_info = None
        self.new_cong_index = None
        self.path_index = None
        self.cong_index = None
        self.cong_types = None

        self.reward_policies = {
            'policy_one': self._get_policy_one,
            'policy_two': self._get_policy_two,
            'policy_three': self._get_policy_three,
            'policy_four': self._get_policy_four,
            'policy_five': self._get_policy_five,
            'policy_six': self._get_policy_six,
            'policy_seven': self._get_policy_seven,
            'policy_eight': self._get_policy_eight,
            'policy_nine': self._get_policy_nine,
            'policy_ten': self._get_policy_ten,
            'policy_eleven': self._get_policy_eleven,
            'policy_twelve': self._get_policy_twelve,
            'policy_thirteen': self._get_policy_thirteen,
            'policy_fourteen': self._get_policy_fourteen,
        }
        # Simulation methods related to routing
        self.routing_obj = Routing(beta=properties['beta'], topology=properties['topology'],
                                   guard_slots=properties['guard_slots'])

    @staticmethod
    def set_seed(seed: int):
        """
        Sets the seed for random generation in the simulation.

        :param seed: The seed to set.
        :type seed: int
        """
        np.random.seed(seed)

    def decay_epsilon(self, amount: float):
        """
        Decays the amount of randomness the agent may experience in its actions.

        :param amount: The amount to decay epsilon by.
        :type amount: float
        """
        self.epsilon -= amount
        if self.curr_episode == 0:
            self.epsilon_stats.append(self.epsilon)
        if self.epsilon < 0.0:
            raise ValueError(f"Epsilon should be greater than 0 but it is {self.epsilon}")

    def _calc_td_averages(self, error_flag):
        matrix = np.array([])
        for episode, curr_list in self.td_errors[error_flag]['errors'].items():
            if episode == '0':
                matrix = np.array([curr_list])
            else:
                matrix = np.vstack((matrix, curr_list))

        self.td_errors[error_flag]['min'] = matrix.min(axis=0, initial=np.inf).tolist()
        self.td_errors[error_flag]['max'] = matrix.max(axis=0, initial=np.inf * -1.0).tolist()
        self.td_errors[error_flag]['average'] = matrix.mean(axis=0).tolist()
        self.td_errors[error_flag].pop('errors')

    def _calc_reward_averages(self, reward_flag):
        matrix = np.array([])
        for episode, curr_list in self.rewards_dict[reward_flag]['rewards'].items():
            if episode == '0':
                matrix = np.array([curr_list])
            else:
                matrix = np.vstack((matrix, curr_list))

        self.rewards_dict[reward_flag]['min'] = matrix.min(axis=0, initial=np.inf).tolist()
        self.rewards_dict[reward_flag]['max'] = matrix.max(axis=0, initial=np.inf * -1.0).tolist()
        self.rewards_dict[reward_flag]['average'] = matrix.mean(axis=0).tolist()
        self.rewards_dict[reward_flag].pop('rewards')

    def _update_stats(self, reward: float, reward_flag: str, td_error: float):
        episode = str(self.curr_episode)

        if episode not in self.rewards_dict[reward_flag]['rewards'].keys():
            self.rewards_dict[reward_flag]['rewards'][episode] = [reward]
            self.td_errors[reward_flag]['errors'][episode] = td_error
            self.sum_rewards[episode] = reward
        else:
            self.sum_rewards[episode] += reward
            self.td_errors[reward_flag]['errors'][episode] += td_error
            self.rewards_dict[reward_flag]['rewards'][episode].append(reward)

        len_rewards = len(self.rewards_dict[reward_flag]['rewards'][episode])
        if self.curr_episode == self.properties['max_iters'] - 1 and len_rewards == self.properties['num_requests']:
            self._calc_reward_averages(reward_flag=reward_flag)
            # self._calc_td_averages(error_flag=reward_flag)

    def save_tables(self):
        """
        Saves trained q-tables.
        """
        if self.sim_type == 'test':
            raise NotImplementedError

        if self.ai_arguments['table_path'] == 'None':
            date_time = f"{self.properties['network']}/{self.properties['date']}/{self.properties['sim_start']}"
            self.ai_arguments['table_path'] = f"{date_time}/{self.properties['thread_num']}"

        create_dir(f"ai/models/q_tables/{self.ai_arguments['table_path']}")
        file_path = f"{os.getcwd()}/ai/models/q_tables/{self.ai_arguments['table_path']}/"
        file_name = f"{self.properties['erlang']}_routes_table_c{self.properties['cores_per_link']}.npy"
        np.save(file_path + file_name, self.q_routes)

        file_name = f"{self.properties['erlang']}_cores_table_c{self.properties['cores_per_link']}.npy"
        np.save(file_path + file_name, self.q_cores)

        properties_dict = {
            'epsilon': self.ai_arguments['epsilon'],
            'episodes': self.properties['max_iters'],
            'learn_rate': self.ai_arguments['learn_rate'],
            'discount_factor': self.ai_arguments['discount'],
            'reward_info': self.rewards_dict,
            'td_info': self.td_errors,
            'epsilon_decay': self.epsilon_stats,
            'sum_rewards': self.sum_rewards,
        }
        file_name = f"{self.properties['erlang']}_params_c{self.properties['cores_per_link']}.json"
        with open(f"{file_path}/{file_name}", 'w', encoding='utf-8') as file:
            json.dump(properties_dict, file)

    def load_tables(self):
        """
        Loads desired previously trained q-tables.
        """
        file_path = f"{os.getcwd()}/ai/models/q_tables/{self.ai_arguments['table_path']}/"
        file_name_one = f"{self.sim_type}_routes_table_c{self.ai_arguments['cores_per_link']}.npy"
        file_name_two = f"{self.sim_type}_cores_table_c{self.ai_arguments['cores_per_link']}.npy"
        try:
            self.q_routes = np.load(file_path + file_name_one)
            self.q_cores = np.load(file_path + file_name_two)
        except FileNotFoundError:
            print('File not found, please ensure if you are testing, an already trained file exists and has been '
                  'specified correctly.')

        file_name = f"{self.properties['erlang']}_params_c{self.properties['cores_per_link']}.json"
        with open(f"{file_path}{file_name}", encoding='utf-8') as file:
            properties_obj = json.load(file)

        self.epsilon = properties_obj['epsilon']
        self.ai_arguments['learn_rate'] = properties_obj['learn_rate']
        self.ai_arguments['discount'] = properties_obj['discount_factor']
        self.rewards_dict = properties_obj['reward_info']

    def _get_policy_fourteen(self, routed: bool):
        if routed:
            reward = 1.0
        else:
            reward = -100.0

        return reward

    def _get_policy_thirteen(self, routed: bool):
        if routed:
            reward = 1.0
        else:
            reward = -10.0

        return reward

    def _get_policy_twelve(self, routed: bool):
        if routed:
            reward = 1.0
        else:
            reward = -1.0

        return reward

    def _get_policy_eleven(self, routed: bool):
        path_cong = find_path_congestion(network_db=self.net_spec_db, path=self.chosen_path)
        if routed:
            reward = 10.0
        else:
            reward = -100.0 * path_cong

        return reward

    def _get_policy_ten(self, routed: bool):
        path_cong = find_path_congestion(network_db=self.net_spec_db, path=self.chosen_path)
        if routed:
            reward = 1.0
        else:
            reward = -100.0 * path_cong

        return reward

    def _get_policy_nine(self, routed: bool):
        core_frag, _ = find_core_frag_cong(net_spec_db=self.net_spec_db, path=self.chosen_path,
                                           core=self.core_index)
        if routed:
            reward = 10.0
        else:
            reward = -100.0 * core_frag

        return reward

    def _get_policy_eight(self, routed: bool):
        core_frag, _ = find_core_frag_cong(net_spec_db=self.net_spec_db, path=self.chosen_path,
                                           core=self.core_index)
        if routed:
            reward = 1.0
        else:
            reward = -100.0 * core_frag

        return reward

    @staticmethod
    def _get_policy_seven(routed: bool):
        if routed:
            reward = 10.0
        else:
            reward = -10.0

        return reward

    @staticmethod
    def _get_policy_six(routed: bool):
        if routed:
            reward = 10.0
        else:
            reward = -1.0

        return reward

    def _get_policy_five(self, routed: bool):
        path_cong = find_path_congestion(network_db=self.net_spec_db, path=self.chosen_path)
        if routed:
            reward = 1.0 - path_cong
        else:
            reward = -1.0

        return reward

    def _get_policy_four(self, routed: bool):
        path_cong = find_path_congestion(network_db=self.net_spec_db, path=self.chosen_path)
        if routed:
            reward = 1.0 - path_cong
        else:
            reward = -10.0

        return reward

    def _get_policy_three(self, routed: bool):
        core_frag, _ = find_core_frag_cong(net_spec_db=self.net_spec_db, path=self.chosen_path,
                                           core=self.core_index)
        if routed:
            reward = 1.0 - core_frag
        else:
            reward = -10.0

        return reward

    def _get_policy_two(self, routed: bool):
        _, core_cong = find_core_frag_cong(net_spec_db=self.net_spec_db, path=self.chosen_path,
                                           core=self.core_index)
        if routed:
            reward = 1.0 - core_cong
        else:
            reward = -10.0

        return reward

    def _get_policy_one(self, routed: bool):
        core_frag, core_cong = find_core_frag_cong(net_spec_db=self.net_spec_db, path=self.chosen_path,
                                                   core=self.core_index)
        if routed:
            cong_reward = 1.0 - core_cong
            frag_reward = 1.0 - core_frag
            reward = cong_reward + frag_reward
        else:
            reward = -10.0

        return reward

    def _get_max_future_q(self, new_cong: float):
        q_values = list()
        self.new_cong_index = self._classify_cong(curr_cong=new_cong)
        path_index, path, _ = self.paths_info[self.path_index]
        self.paths_info[self.path_index] = (path_index, path, self.new_cong_index)

        for path_index, _, cong_index in self.paths_info:
            curr_q = self.q_routes[self.source][self.destination][path_index][cong_index]['q_value']
            q_values.append(curr_q)

        return np.max(q_values)

    def update_env(self, routed: bool):
        """
        Controls the updating of routing and core assignment q-tables.

        :param routed: Whether the request was routed or not.
        :type routed: bool
        """
        self._update_routes_q_values(routed)
        self._update_cores_q_values(routed)

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

                    for c_index, _ in enumerate(self.cong_types):
                        self.q_routes[source, destination, k, c_index] = (curr_path, 0.0)

                        for core_action in range(self.num_cores):
                            self.q_cores[source, destination, k, c_index, core_action] = (curr_path, core_action, 0.0)

    def setup_env(self):
        """
        Sets up the environment (q-tables) for the q-learning algorithm.
        """
        self.epsilon = self.ai_arguments['epsilon']
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

    def _assign_congestion(self, paths):
        resp = list()
        for path_index, curr_path in enumerate(paths):
            curr_cong = find_path_congestion(path=curr_path, network_db=self.net_spec_db)
            cong_index = self._classify_cong(curr_cong=curr_cong)

            resp.append((path_index, curr_path, cong_index))

        return resp

    def route(self):
        """
        Provide a route for a given request based on the q-learning algorithm.

        :return: A route for the request.
        :rtype: List
        """
        random_float = np.round(np.random.uniform(0, 1), decimals=1)
        paths = self.q_routes[self.source][self.destination]['path'][:, 0]
        self.paths_info = self._assign_congestion(paths=paths)

        if random_float < self.epsilon:
            self.path_index = np.random.choice(self.k_paths)
            self.cong_index = self.paths_info[self.path_index][-1]
            self.chosen_path = self.q_routes[self.source][self.destination][self.path_index][self.cong_index]['path']
        else:
            best_path = self._get_max_q(paths=self.paths_info)
            self.path_index, self.chosen_path, self.cong_index = best_path

        if len(self.chosen_path) == 0:
            raise ValueError('The chosen path can not be None')

        return self.chosen_path

    def core_assignment(self):
        """
        Find the appropriate core assignment for a given request.

        :return: A core index
        :rtype: int
        """
        random_float = np.round(np.random.uniform(0, 1), decimals=1)

        if random_float < self.epsilon:
            self.core_index = np.random.randint(0, self.properties['cores_per_link'])
        else:
            q_values = self.q_cores[self.source][self.destination][self.path_index][self.cong_index]['q_value']
            self.core_index = np.argmax(q_values)

        return self.core_index
