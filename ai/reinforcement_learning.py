# Standard library imports
import os
import json

# Third party imports
import numpy as np
import networkx as nx

# Local application imports
from useful_functions.sim_functions import get_path_mod, find_path_len
from useful_functions.handle_dirs_files import create_dir


# TODO: Write tests for these methods and check sim scripts for other tests as well
# TODO: Better reward scheme
# TODO: Tune hyper-params
# TODO: Significant plots


class QLearning:
    """
    Controls methods related to the Q-learning reinforcement learning algorithm.
    """

    def __init__(self, epsilon: float = 0.2, episodes: int = 1000, learn_rate: float = 0.5,
                 discount: float = 0.2, topology: nx.Graph = None):
        """
        Initializes the QLearning class.

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
        # Contains all state and action value pairs
        self.q_table = None
        self.epsilon = epsilon
        # TODO: How should we be using episodes?
        self.episodes = episodes
        self.learn_rate = learn_rate
        self.discount = discount
        self.topology = topology

        # Statistics used for plotting
        self.rewards_dict = {'episode': [], 'average': [], 'min': [], 'max': [], 'rewards': []}

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

    def plot_rewards(self):
        """
        Plots reward values vs. episodes.
        """
        raise NotImplementedError

    def save_table(self, path: str, max_segments: int):
        """
        Saves the current Q-table and hyperparameters used to create it.

        :param path: The path for the table to be saved.
        :type path: str

        :param max_segments: The number of light segments allowed for a single request.
        :type max_segments: int
        """
        create_dir(f'ai/q_tables/{path}')

        with open(f'{os.getcwd()}/ai/q_tables/{path}/trained_table_ls{max_segments}.npy', 'wb') as f:
            np.save(f, self.q_table)

        params_dict = {
            'epsilon': self.epsilon,
            'episodes': self.episodes,
            'learn_rate': self.learn_rate,
            'discount_factor': self.discount,
            'reward_info': self.rewards_dict
        }
        with open(f'{os.getcwd()}/ai/q_tables/{path}/hyper_params_ls{max_segments}.json', 'w') as f:
            json.dump(params_dict, f)

    def load_table(self, path: str, max_segments: int):
        """
        Loads a saved Q-table and previous hyperparameters.

        :param path: The path for the table to be loaded.
        :type path: str

        :param max_segments: The number of light segments allowed for a single request.
        :type max_segments: int
        """
        self.q_table = np.load(f'{os.getcwd()}/ai/q_tables/{path}/trained_table_ls{max_segments}.npy')

        with open(f'{os.getcwd()}/ai/q_tables/{path}/hyper_params_ls{max_segments}.json') as f:
            params_obj = json.load(f)
            self.epsilon = params_obj['epsilon']
            self.learn_rate = params_obj['learn_rate']
            self.discount = params_obj['discount_factor']

    def update_environment(self, routed: bool, path: list, free_slots: int):
        """
        The custom environment that updates the Q-table with respect to a reward policy.

        :param routed: If a request was successfully routed or not.
        :type routed: bool

        :param path: The path taken from source to destination.
        :type path: list

        :param free_slots: The total number of available slots after request allocation in the path.
        :type free_slots: int

        :return: The reward value.
        :rtype: int
        """
        reward = 0
        if not isinstance(free_slots, bool):
            # Penalize by path length and number of nodes
            path_len = float(find_path_len(path=path, topology=self.topology))
            num_nodes = float(len(path))

            if routed:
                reward += ((100.0 * float(free_slots)) - path_len - num_nodes)
            else:
                reward -= (1000.0 - path_len - num_nodes)
        else:
            # No path found due to the Q-table
            reward -= 10000.0

        for i in range(len(path) - 1):
            state = int(path[i])
            new_state = int(path[i + 1])

            # If terminating node, only consider the reward (there is no future state)
            if i + 1 == len(path):
                max_future_q = reward
            else:
                max_future_q = np.nanargmax(self.q_table[new_state])

            current_q = self.q_table[(state, new_state)]
            new_q = ((1.0 - self.learn_rate) * current_q) + (
                    self.learn_rate * (reward + self.discount * max_future_q))

            self.q_table[(state, new_state)] = new_q

    def setup_environment(self):
        """
        Initializes the environment.
        """
        total_nodes = self.topology.number_of_nodes()
        # Init q-table for USNet, a 24 node network
        self.q_table = np.zeros((total_nodes, total_nodes))

        for source in range(0, total_nodes):
            for destination in range(0, total_nodes):
                if source == destination:
                    self.q_table[(source, destination)] = np.nan
                    continue

                # A link exists between these nodes, init to zero
                if str(source) in self.topology.neighbors(str(destination)):
                    self.q_table[(source, destination)] = 0
                else:
                    self.q_table[(source, destination)] = np.nan

    def _sort_q_values(self, last_node: int):
        """
        Given a row in the Q-table, sort the indexes based on their corresponding Q-values.

        :param last_node: The last node chosen in the path, aka the current row.
        :type last_node: int

        :return: The sorted indexes in the row based on their Q-values.
        :rtype: list
        """
        array_to_sort = self.q_table[last_node]
        # Create a mask to ignore nan values
        mask = ~np.isnan(array_to_sort)
        # Sort ignoring non-nan values but keeping original indexes
        sorted_indexes = np.argsort(-array_to_sort[mask])
        sorted_original_indexes = np.where(mask)[0][sorted_indexes]

        return sorted_original_indexes

    @staticmethod
    def _find_next_best(path: list, indexes: list):
        """
        Given a node already exists in a path, attempt to find the next best one.

        :param path: The nodes already chosen in a single path.
        :type path: list

        :param indexes: The sorted indexes of the descending Q-values in the Q-table.
        :type indexes: list

        :return: The next node or a boolean if no nodes could be found.
        :rtype: int or bool
        """
        curr_index = 0
        while True:
            try:
                next_node = indexes[curr_index + 1]
            except IndexError:
                return False

            if str(next_node) in path:
                curr_index = curr_index + 1
            else:
                return next_node

    def route(self, source: int, destination: int, mod_formats: dict):
        """
        Determines a route from source to destination using Q-Learning.

        :param source: The source node.
        :type source: int

        :param destination: The destination node.
        :type destination: int

        :param mod_formats: Modulation formats for a selected bandwidth and their potential reach.
        :type mod_formats: dict

        :return: The path from source to destination.
        :rtype: list
        """
        while True:
            path = [source]
            last_node = int(source)
            while True:
                # Choose a random action with respect to epsilon
                random_float = np.round(np.random.uniform(0, 1), decimals=1)
                if random_float < self.epsilon:
                    # Only generate a random number for links that exists
                    valid_indexes = np.where(~np.isnan(self.q_table[last_node]))[0]
                    chosen_index = np.random.randint(len(valid_indexes))

                    next_node = valid_indexes[chosen_index]
                    random_node = True
                else:
                    # Sorted Q-values in a single row
                    sorted_values = self._sort_q_values(last_node=last_node)
                    next_node = sorted_values[0]
                    random_node = False

                q_value = self.q_table[(last_node, next_node)]
                # Node has already been chosen, force to choose again for the time being
                if str(next_node) in path:
                    # Attempt to choose another node
                    # TODO: This may get stuck again, choosing the same random node over and over
                    if random_node:
                        continue
                    # Try to assign the next best node
                    else:
                        next_node = self._find_next_best(path=path, indexes=sorted_values)
                        # Blocking caused by the Q-table, penalize and try again
                        if not next_node:
                            self.update_environment(routed=False, path=path, free_slots=False)
                            break

                path.append(str(next_node))
                last_node = next_node
                if str(next_node) == destination:
                    path_len = find_path_len(path, self.topology)
                    mod_format = get_path_mod(mod_formats, path_len)
                    return path, mod_format


if __name__ == '__main__':
    pass
