# Standard library imports
import os
import json
from itertools import permutations

# Third party imports
import numpy as np
import networkx as nx

# Local application imports
from useful_functions.handle_dirs_files import create_dir


class QLearning:
    """
    Controls methods related to the Q-learning reinforcement learning algorithm.
    """

    def __init__(self, is_training: bool = None, epsilon: float = 0.2, episodes: int = None, learn_rate: float = 0.9,
                 discount: float = 0.9, topology: nx.Graph = None, k_paths: int = 1):
        """
        Initializes the QLearning class.

        :param is_training: A flag to tell us to save and load a trained table or one that was used for testing.
        :type is_training: bool

        :param epsilon: Parameter in the bellman equation to determine degree of randomness.
        :type epsilon: float

        :param episodes: The number of iterations or simulations.
        :type episodes: int

        :param learn_rate: The learning rate, alpha, in the bellman equation.
        :type learn_rate: float

        :param discount: The discount factor in the bellman equation to determine the balance between current and
                         future rewards.
        :type discount: float

        :param topology: The network topology.
        :type: topology: nx.Graph

        :param k_paths: The number of K shortest paths to consider in the Q-table.
        :type k_paths: int
        """
        if is_training:
            self.sim_type = 'train'
        else:
            self.sim_type = 'test'
        # Contains all state and action value pairs
        self.q_table = None
        self.epsilon = epsilon
        self.episodes = episodes
        self.learn_rate = learn_rate
        self.discount = discount
        self.topology = topology
        self.k_paths = k_paths

        # Statistics used for plotting
        self.rewards_dict = {'average': [], 'min': [], 'max': [], 'rewards': []}
        # The last chosen path, used for ease of indexing/searching the q-table
        self.last_chosen = None

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
        self.epsilon -= amount
        if self.epsilon < 0.0:
            raise ValueError(f'Epsilon should be greater than 0 but it is {self.epsilon}')

    def _update_rewards_dict(self, reward: float):
        """
        Updates the reward dictionary with desired information related to the Q-learning algorithm.

        :param reward: The reword achieved for a single episode.
        :type reward: float
        """
        self.rewards_dict['rewards'].append(reward)
        self.rewards_dict['min'] = min(self.rewards_dict['rewards'])
        self.rewards_dict['max'] = max(self.rewards_dict['rewards'])
        self.rewards_dict['average'] = sum(self.rewards_dict['rewards']) / float(len(self.rewards_dict['rewards']))

    def save_table(self, path: str, max_segments: int, cores_per_link: int):
        """
        Saves the current Q-table and hyperparameters used to create it.

        :param path: The path for the table to be saved.
        :type path: str

        :param max_segments: The number of light segments allowed for a single request.
        :type max_segments: int

        :param cores_per_link: The number of optical fiber cores on a single link.
        :type cores_per_link: int
        """
        create_dir(f'ai/q_tables/{path}')

        with open(f'{os.getcwd()}/ai/q_tables/{path}/{self.sim_type}_table_ls{max_segments}_c{cores_per_link}.json',
                  'w', encoding='utf-8') as file:
            json.dump(self.q_table, file)

        params_dict = {
            'epsilon': self.epsilon,
            'episodes': self.episodes,
            'learn_rate': self.learn_rate,
            'discount_factor': self.discount,
            'reward_info': self.rewards_dict
        }
        with open(f'{os.getcwd()}/ai/q_tables/{path}/hyper_params_ls{max_segments}_c{cores_per_link}.json', 'w',
                  encoding='utf-8') as file:
            json.dump(params_dict, file)

    def load_table(self, path: str, max_segments: int, cores_per_link: int):
        """
        Loads a saved Q-table and previous hyperparameters.

        :param path: The path for the table to be loaded.
        :type path: str

        :param max_segments: The number of light segments allowed for a single request.
        :type max_segments: int

        :param cores_per_link: The number of optical fiber cores on a single link.
        :type cores_per_link: int
        """
        try:
            with open(
                    f'{os.getcwd()}/ai/q_tables/{path}/{self.sim_type}_table_ls{max_segments}_c{cores_per_link}.json',
                    encoding='utf-8') as file:
                self.q_table = json.load(file)
        except FileNotFoundError:
            print('File not found, please ensure if you are testing, an already trained file exists and has been '
                  'specified correctly.')

        with open(f'{os.getcwd()}/ai/q_tables/{path}/hyper_params_ls{max_segments}_c{cores_per_link}.json',
                  encoding='utf-8') as file:
            params_obj = json.load(file)
            self.epsilon = params_obj['epsilon']
            self.learn_rate = params_obj['learn_rate']
            self.discount = params_obj['discount_factor']

    def update_environment(self, routed: bool, path: list, num_segments: int, free_slots: int):
        """
        The custom environment that updates the Q-table with respect to a reward policy.

        :param routed: If a request was successfully routed or not.
        :type routed: bool

        :param path: The list of nodes from source to destination.
        :type path: list

        :param num_segments: The number of segments that this request was "sliced" into.
        :type num_segments: int

        :param free_slots: The total number of free slots on the selected path.
        :type free_slots: int

        :return: The reward value.
        :rtype: int
        """
        source = int(path[0])
        destination = int(path[-1])
        # TODO: LS = 1 we want the shortest path possible...something to consider
        reward = 0
        if routed:
            if free_slots == 0:
                cong_percent = 100.0
            else:
                # TODO: Hard coded (number of slots per link is 128) for now
                cong_percent = ((128.0 * float(len(path))) / float(free_slots)) * 100.0
            cong_quality = 100.0 - cong_percent
            seg_quality = 100 / num_segments
            reward += ((cong_quality + seg_quality) / 200) * 100
        else:
            reward -= 1800.0

        max_future_q = max(  # pylint: disable=consider-using-generator
            [lst[1] for lst in self.q_table[source][destination]])
        current_q = self.q_table[source][destination][self.last_chosen][1]
        new_q = ((1.0 - self.learn_rate) * current_q) + (
                self.learn_rate * (reward + self.discount * max_future_q))

        self.q_table[source][destination][self.last_chosen][1] = new_q

        self._update_rewards_dict(reward=reward)

    def setup_environment(self):
        """
        Initializes the environment.
        """
        nodes = list(self.topology.nodes())
        # Get all combinations of source and destination nodes
        combinations_list = list(permutations(nodes, 2))

        self.q_table = [[[0 for _ in range(self.k_paths)] for _ in range(len(nodes))] for _ in range(len(nodes))]

        for source, destination in combinations_list:
            shortest_paths = nx.shortest_simple_paths(self.topology, source=source, target=destination, weight='length')
            for i, path in enumerate(shortest_paths):
                if i == self.k_paths:
                    break
                # Assign a q-value for every kth path
                self.q_table[int(source)][int(destination)][i] = [path, 0]

    def route(self, source: int, destination: int):
        """
        Determines a route from source to destination using Q-Learning.

        :param source: The source node.
        :type source: int

        :param destination: The destination node.
        :type destination: int

        :return: The path from source to destination.
        :rtype: list
        """
        random_float = np.round(np.random.uniform(0, 1), decimals=1)
        if random_float < self.epsilon:
            chosen_path = np.random.randint(self.k_paths)
        else:
            chosen_path = max(index for index, value in enumerate(self.q_table[source][destination]))

        self.last_chosen = chosen_path
        return self.q_table[source][destination][chosen_path][0]


if __name__ == '__main__':
    pass
