# Standard library imports
import os
import json

# Third party imports
import numpy as np
import networkx as nx

# Local application imports
from useful_functions.handle_dirs_files import create_dir
from useful_functions.sim_functions import find_max_length, find_path_len
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
        self.q_table = None
        # Statistics to evaluate our reward function
        self.rewards_dict = {'average': [], 'min': [], 'max': [], 'rewards': {}}

        self.curr_episode = None
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
        self.reward_policies = {
            'baseline': self._get_baseline_reward,
            'policy_one': self._get_policy_one,
        }
        # Simulation methods related to routing
        self.routing_obj = Routing(beta=properties['beta'], topology=properties['topology'],
                                   guard_slots=properties['guard_slots'])

    @staticmethod
    def set_seed(seed: int):
        """
        Used to set the seed for controlling 'random' generation.

        :param seed: The seed to be set for numpy random generation.
        :type seed: int
        """
        np.random.seed(seed)

    def decay_epsilon(self, amount: float):
        """
        Decays our epsilon value by a specified amount.

        :param amount: The amount to decay epsilon by.
        :type amount: float
        """
        self.ai_arguments['epsilon'] -= amount
        if self.ai_arguments['epsilon'] < 0.0:
            raise ValueError(f"Epsilon should be greater than 0 but it is {self.ai_arguments['epsilon']}")

    def _update_rewards_dict(self, reward: float = None):
        """
        Updates the reward dictionary for plotting purposes later on.

        :param reward: The numerical reward in the last episode.
        :type reward: float
        """
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
        """
        Saves the current Q-table to a desired path.
        """
        if self.sim_type == 'test':
            raise NotImplementedError

        if self.ai_arguments['table_path'] == 'None':
            date_time = f"{self.properties['network']}/{self.properties['date']}/{self.properties['sim_start']}"
            self.ai_arguments['table_path'] = f"{date_time}/{self.properties['thread_num']}"

        create_dir(f"ai/models/q_tables/{self.ai_arguments['table_path']}")
        file_path = f"{os.getcwd()}/ai/models/q_tables/{self.ai_arguments['table_path']}/"
        file_name = f"{self.properties['erlang']}_table_c{self.properties['cores_per_link']}.npy"
        np.save(file_path + file_name, self.q_table)

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
        """
        Loads a previously trained Q-table.
        """
        file_path = f"{os.getcwd()}/ai/models/q_tables/{self.ai_arguments['table_path']}/"
        file_name = f"{self.sim_type}_table_c{self.ai_arguments['cores_per_link']}.npy"
        try:
            self.q_table = np.load(file_path + file_name)
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

    def _path_cost(self):
        max_length = find_max_length(source=self.chosen_path[0], destination=self.chosen_path[-1],
                                     topology=self.properties['topology'])

        return max_length

    @staticmethod
    def _get_baseline_reward(routed: bool, path_mod: str):  # pylint: disable=unused-argument
        return 1.0 if routed else -1.0

    def _get_policy_one(self, routed: bool, path_mod: str):
        if routed:
            longest_len = self._path_cost()
            bandwidth_obj = self.properties['mod_per_bw'][self.chosen_bw]
            slots_used = float(bandwidth_obj[path_mod]['slots_needed'])
            max_slots = float(max(item['slots_needed'] for item in bandwidth_obj.values()))
            q_term_one = slots_used / max_slots

            path_len = find_path_len(path=self.chosen_path, topology=self.properties['topology'])
            q_term_two = path_len / longest_len

            return 3.0 - q_term_one - q_term_two

        return -20.0

    def update_environment(self, routed: bool, spectrum: dict, path_mod: str):  # pylint: disable=unused-argument
        """
        Updates the Q-learning environment.

        :param routed: Whether the path chosen was successfully routed or not.
        :type routed: bool

        :param spectrum: Relevant information regarding the spectrum of the current request.
        :type spectrum: dict
        """
        policy = self.ai_arguments.get('policy')
        if policy not in self.reward_policies:
            raise NotImplementedError('Reward policy not recognized.')

        reward = self.reward_policies[policy](routed=routed, path_mod=path_mod)
        self._update_rewards_dict(reward=reward)

        max_future_q = float('-inf')
        current_q = None
        path_index = None
        source = int(self.chosen_path[0])
        destination = int(self.chosen_path[-1])
        for index, matrix in enumerate(self.q_table[source][destination]):
            path = matrix[0]
            curr_value = matrix[1]

            if path == self.chosen_path:
                path_index = index
                current_q = curr_value

            if curr_value > max_future_q:
                max_future_q = curr_value

        new_q = ((1.0 - self.ai_arguments['learn_rate']) * current_q) + (
                self.ai_arguments['learn_rate'] * (reward + (self.ai_arguments['discount'] * max_future_q)))

        self.q_table[source][destination][path_index][1] = new_q

    def setup_environment(self):
        """
        Initializes the environment i.e., the Q-table.
        """
        num_nodes = len(list(self.properties['topology'].nodes()))
        self.k_paths = self.properties['k_paths']
        self.q_table = [[[] for _ in range(num_nodes)] for _ in range(num_nodes)]

        for source in range(0, num_nodes):
            for destination in range(0, num_nodes):
                # A node cannot be attached to itself
                if source == destination:
                    continue

                # A link exists between these two nodes
                shortest_paths = nx.shortest_simple_paths(self.properties['topology'], str(source), str(destination))
                for k, curr_path in enumerate(shortest_paths):
                    if k >= self.k_paths:
                        break
                    # Initial Q-values set to zero
                    self.q_table[source][destination].append([curr_path, 0])

    def route(self):
        """
        Based on the Q-learning algorithm, find a route for any given request.
        """
        random_float = np.round(np.random.uniform(0, 1), decimals=1)
        if random_float < self.ai_arguments['epsilon']:
            random_path = np.random.choice(self.k_paths)
            # Always zero indexes because the next index is the corresponding q-value for that path
            self.chosen_path = self.q_table[self.source][self.destination][random_path][0]
        else:
            max_value = float('-inf')
            # Navigate the nested list
            for outer_list in self.q_table[self.source][self.destination]:
                if outer_list[1] > max_value:
                    max_value = outer_list[1]
                    self.chosen_path = outer_list[0]

        if len(self.chosen_path) == 0:
            raise ValueError('The chosen path can not be None')

        return self.chosen_path
