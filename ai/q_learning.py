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

        # TODO: Make sure these values are updated everytime in sim_scripts
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
        self.rewards_dict['rewards'].append(reward)
        self.rewards_dict['min'] = min(self.rewards_dict['rewards'])
        self.rewards_dict['max'] = max(self.rewards_dict['rewards'])
        self.rewards_dict['average'] = sum(self.rewards_dict['rewards']) / float(len(self.rewards_dict['rewards']))

    def save_table(self, path, cores_per_link):
        create_dir(f'ai/q_tables/{path}')

        with open(f'{os.getcwd()}/ai/q_tables/{path}/{self.sim_type}_table_c{cores_per_link}.json',
                  'w', encoding='utf-8') as file:
            json.dump(self.q_table, file)

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
            with open(
                    f'{os.getcwd()}/ai/q_tables/{path}/{self.sim_type}_table_c{cores_per_link}.json',
                    encoding='utf-8') as file:
                self.q_table = json.load(file)
        except FileNotFoundError:
            print('File not found, please ensure if you are testing, an already trained file exists and has been '
                  'specified correctly.')

        with open(f'{os.getcwd()}/ai/q_tables/{path}/hyper_params_c{cores_per_link}.json',
                  encoding='utf-8') as file:
            params_obj = json.load(file)
            self.epsilon = params_obj['epsilon']
            self.learn_rate = params_obj['learn_rate']
            self.discount = params_obj['discount_factor']

    def update_environment(self):
        # TODO: Update
        reward = -10000000
        # TODO: Penalty for no path found due to Q-learning?
        for i in range(len(self.chosen_path) - 1):
            # Source node, current state
            state = int(self.chosen_path[i])
            # Destination node, new state
            new_state = int(self.chosen_path[i + 1])

            # The terminating node, there is no future state
            # TODO: Update
            if i + 1 == len(self.chosen_path):
                max_future_q = None
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

    def _find_next_node(self):
        # TODO: Update
        some_node = None
        array_to_sort = self.q_table[some_node]
        # Create a mask to ignore nan values
        mask = ~np.isnan(array_to_sort)
        # Sort ignoring non-nan values but keeping original indexes
        sorted_indexes = np.argsort(-array_to_sort[mask])
        sorted_original_indexes = np.where(mask)[0][sorted_indexes]
        next_node = None

        for i in range(len(sorted_original_indexes)):
            if next_node not in self.chosen_path:
                self.chosen_path.append(next_node)
                return True
            else:
                continue

        return False

    def route(self):
        self.chosen_path = [self.source]
        # TODO: Nodes equal to the values from the q-table for the source entry
        #   - Sort them from greatest to least, pick each of them and if you make the end of the list, block
        # TODO: These need to be indexes
        nodes = self.q_table[self.source]

        while True:
            # TODO: This random probably has to be fixed, (e.g., not actually 10 percent randomness)
            #   - Need a limit on randomness
            random_float = np.round(np.random.uniform(0, 1), decimals=1)
            if random_float < self.epsilon:
                # TODO: Check on this length, either length of nodes minus one or not?
                # TODO: This may get stuck here
                # TODO: Also need to check if there's and actual connection (is not nan)
                next_node = np.random.randint(len(nodes))
            else:
                # TODO: This is dependent on the structure of the Q-table, most likely needs to be updated
                next_node = self._find_next_node(nodes[0])

            # TODO: Also need to check for np.nan values
            if next_node is not False:
                if next_node == self.destination:
                    return

                nodes = self.q_table[next_node]
            # TODO: Here is where you need to check for no more paths found
            else:
                continue

        return self.chosen_path


if __name__ == '__main__':
    pass
