# Standard library imports
import os
import json
from itertools import permutations

# Third party imports
import numpy as np
import networkx as nx

# Local application imports
from useful_functions.handle_dirs_files import create_dir
from sim_scripts.routing import Routing


class QLearning:
    """
    Controls methods related to the Q-learning reinforcement learning algorithm.
    """

    def __init__(self, is_training: bool = None, epsilon: float = 0.2, episodes: int = None, learn_rate: float = 0.9,
                 discount: float = 0.9, topology: nx.Graph = None):
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

        # Statistics used for plotting
        self.rewards_dict = {'average': [], 'min': [], 'max': [], 'rewards': []}

        self.source = None
        self.destination = None
        self.chosen_path = None

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
        # TODO: This method isn't getting called
        self.rewards_dict['rewards'].append(reward)
        self.rewards_dict['min'] = min(self.rewards_dict['rewards'])
        self.rewards_dict['max'] = max(self.rewards_dict['rewards'])
        self.rewards_dict['average'] = sum(self.rewards_dict['rewards']) / float(len(self.rewards_dict['rewards']))

    def save_table(self, path, cores_per_link):
        create_dir(f'ai/q_tables/{path}')

        file_path = f'{os.getcwd()}/ai/q_tables/{path}/{self.sim_type}_table_c{cores_per_link}.npy'
        np.save(file_path, self.q_table)

        params_dict = {
            'epsilon': self.epsilon,
            'episodes': self.episodes,
            'learn_rate': self.learn_rate,
            'discount_factor': self.discount,
            'reward_info': self.rewards_dict
        }
        with open(f'{os.getcwd()}/ai/q_tables/{path}/hyper_params_c{cores_per_link}.json', 'w',
                  encoding='utf-8') as file:
            json.dump(params_dict, file)

    def load_table(self, path, cores_per_link):
        try:
            file_path = f'{os.getcwd()}/ai/q_tables/{path}/{self.sim_type}_table_c{cores_per_link}.npy'
            self.q_table = np.load(file_path)
        except FileNotFoundError:
            print('File not found, please ensure if you are testing, an already trained file exists and has been '
                  'specified correctly.')

        with open(f'{os.getcwd()}/ai/q_tables/{path}/hyper_params_c{cores_per_link}.json',
                  encoding='utf-8') as file:
            params_obj = json.load(file)
            self.epsilon = params_obj['epsilon']
            self.learn_rate = params_obj['learn_rate']
            self.discount = params_obj['discount_factor']

    # TODO: Need to create a method that just calculates the NLI cost of a path
    def _get_nli_cost(self):
        pass

    def update_environment(self, routed):
        # TODO: Update
        nli_cost = self._get_nli_cost()
        if not routed:
            reward = -10000
        else:
            reward = 100
        for i in range(len(self.chosen_path) - 1):
            # Source node, current state
            state = int(self.chosen_path[i])
            # Destination node, new state
            new_state = int(self.chosen_path[i + 1])

            # The terminating node, there is no future state (equal to reward for now)
            if i + 1 == len(self.chosen_path):
                max_future_q = reward
            else:
                max_future_q = np.nanargmax(self.q_table[new_state])

            current_q = self.q_table[(state, new_state)]
            new_q = ((1.0 - self.learn_rate) * current_q) + (
                    self.learn_rate * (reward + self.discount * max_future_q))

            self.q_table[(state, new_state)] = new_q

    def setup_environment(self):
        num_nodes = len(list(self.topology.nodes()))
        self.q_table = np.zeros((num_nodes, num_nodes))

        for source in range(0, num_nodes):
            for destination in range(0, num_nodes):
                # A node cannot be attached to itself
                if source == destination:
                    self.q_table[(source, destination)] = np.nan
                    continue

                # A link exists between these two nodes
                if str(source) in self.topology.neighbors((str(destination))):
                    self.q_table[(source, destination)] = 0
                else:
                    self.q_table[(source, destination)] = np.nan

    def _find_next_node(self, curr_node):
        array_to_sort = self.q_table[curr_node]
        # Create a mask to ignore nan values
        mask = ~np.isnan(array_to_sort)
        # Sort ignoring non-nan values but keeping original indexes
        sorted_indexes = np.argsort(-array_to_sort[mask])
        sorted_original_indexes = np.where(mask)[0][sorted_indexes]

        for next_node in sorted_original_indexes:
            if str(next_node) not in self.chosen_path:
                self.chosen_path.append(str(next_node))
                return True, next_node
            else:
                continue

        return False, None

    def route(self):
        self.chosen_path = [str(self.source)]
        nodes = self.q_table[self.source]

        curr_node = self.source
        while True:
            random_float = np.round(np.random.uniform(0, 1), decimals=1)
            if random_float < self.epsilon:
                next_node = np.random.randint(len(nodes))
                if str(next_node) in self.chosen_path or np.isnan(self.q_table[(curr_node, next_node)]):
                    continue
                self.chosen_path.append(str(next_node))
                found_next = True
            else:
                found_next, next_node = self._find_next_node(curr_node)

            if found_next is not False:
                if next_node == self.destination:
                    return self.chosen_path

                curr_node = next_node
                nodes = self.q_table[next_node]
                continue

            # Q-routing chose too many nodes, no path found due to Q-routing constraint
            return False


if __name__ == '__main__':
    pass
