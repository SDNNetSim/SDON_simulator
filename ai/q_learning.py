# Standard library imports
import os
import json

# Third party imports
import numpy as np

# Local application imports
from useful_functions.handle_dirs_files import create_dir
from useful_functions.sim_functions import get_path_mod, find_path_len
from sim_scripts.routing import Routing


class QLearning:
    """
    Controls methods related to the Q-learning reinforcement learning algorithm.
    """

    def __init__(self, params: dict):
        """
        Initializes the QLearning class.
        """
        if params['is_training']:
            self.sim_type = 'train'
        else:
            self.sim_type = 'test'
        # Contains all state and action value pairs
        self.q_table = None
        self.epsilon = params['epsilon']
        self.episodes = params['episodes']
        self.learn_rate = params['learn_rate']
        self.discount = params['discount']
        self.topology = params['topology']
        self.table_path = params['table_path']
        self.cores_per_link = params['cores_per_link']
        self.curr_episode = params['curr_episode']
        self.mod_per_bw = params['mod_per_bw']

        # Statistics to evaluate our reward function
        self.rewards_dict = {'average': [], 'min': [], 'max': [], 'rewards': {}}

        # Source node, destination node, and the resulting path
        self.source = None
        self.destination = None
        self.chosen_path = None
        # The chosen bandwidth for the current request
        self.chosen_bw = None
        # The NLI cost for a given path
        self.nli_cost = None
        # The latest up-to-date network spectrum database
        self.net_spec_db = None
        # Simulation methods related to routing
        self.routing_obj = Routing(beta=params['beta'], topology=self.topology)

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

        if self.curr_episode == self.episodes - 1:
            self.rewards_dict['min'] = min(self.rewards_dict['rewards'][episode])
            self.rewards_dict['max'] = max(self.rewards_dict['rewards'][episode])
            self.rewards_dict['average'] = sum(self.rewards_dict['rewards'][episode]) / float(
                len(self.rewards_dict['rewards'][episode]))

    def save_table(self):
        """
        Saves the current Q-table to a desired path.
        """
        create_dir(f'ai/q_tables/{self.table_path}')

        file_path = f'{os.getcwd()}/ai/q_tables/{self.table_path}/{self.sim_type}_table_c{self.cores_per_link}.npy'
        np.save(file_path, self.q_table)

        params_dict = {
            'epsilon': self.epsilon,
            'episodes': self.episodes,
            'learn_rate': self.learn_rate,
            'discount_factor': self.discount,
            'reward_info': self.rewards_dict
        }
        with open(f'{os.getcwd()}/ai/q_tables/{self.table_path}/hyper_params_c{self.cores_per_link}.json', 'w',
                  encoding='utf-8') as file:
            json.dump(params_dict, file)

    def load_table(self):
        """
        Loads a previously trained Q-table.
        """
        try:
            file_path = f'{os.getcwd()}/ai/q_tables/{self.table_path}/{self.sim_type}_table_c{self.cores_per_link}.npy'
            self.q_table = np.load(file_path)
        except FileNotFoundError:
            print('File not found, please ensure if you are testing, an already trained file exists and has been '
                  'specified correctly.')

        with open(f'{os.getcwd()}/ai/q_tables/{self.table_path}/hyper_params_c{self.cores_per_link}.json',
                  encoding='utf-8') as file:
            params_obj = json.load(file)
            self.epsilon = params_obj['epsilon']
            self.learn_rate = params_obj['learn_rate']
            self.discount = params_obj['discount_factor']
            self.rewards_dict = params_obj['reward_info']

    def _get_nli_cost(self):
        """
        Uses the routing object to get the non-linear impairment cost from the selected path.
        """
        mod_formats = self.mod_per_bw[self.chosen_bw]
        path_len = find_path_len(self.chosen_path, self.topology)
        mod_format = get_path_mod(mod_formats, path_len)

        self.routing_obj.net_spec_db = self.net_spec_db
        self.routing_obj.slots_needed = self.mod_per_bw[self.chosen_bw][mod_format]['slots_needed']
        self.nli_cost = self.routing_obj.nli_path(path=self.chosen_path)

    def update_environment(self, routed: bool):
        """
        Updates the Q-learning environment.

        :param routed: Whether the path chosen was successfully routed or not.
        :type routed: bool
        """
        # TODO: Update reward
        #   - Need worst case NLI (Ask)
        nli_worst = 100
        if not routed:
            reward = -1.0
        else:
            reward = 1.0 - (self.nli_cost / nli_worst)

        self._update_rewards_dict(reward=reward)

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
        """
        Initializes the environment i.e., the Q-table.
        """
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

    def _find_next_node(self, curr_node: int):
        """
        Sort through the Q-values in a given row (current node) and choose the best one. Note that we can't always
        choose the maximum Q-value since we do not allow a node to be in the same path more than once.

        :param curr_node: The current node number.
        :type curr_node: int
        """
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

        return False, None

    def route(self):
        """
        Based on the Q-learning algorithm, find a route for any given request.
        """
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
                    self._get_nli_cost()

                    return self.chosen_path

                curr_node = next_node
                nodes = self.q_table[next_node]
                continue

            # Q-routing chose too many nodes, no path found due to Q-routing constraint
            return False
