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

    def __init__(self, properties: dict):
        """
        Initializes the QLearning class.
        """
        ai_arguments = properties['ai_arguments']
        if ai_arguments['is_training']:
            self.sim_type = 'train'
        else:
            self.sim_type = 'test'
        # Contains all state and action value pairs
        self.q_table = None
        # TODO: Try to just use properties and not all these variables
        self.epsilon = ai_arguments['epsilon']
        self.episodes = properties['max_iters']
        self.learn_rate = ai_arguments['learn_rate']
        self.discount = ai_arguments['discount']
        self.table_path = ai_arguments['table_path']
        self.cores_per_link = properties['cores_per_link']
        self.mod_per_bw = properties['mod_per_bw']
        self.guard_slots = properties['guard_slots']
        self.topology = properties['topology']

        # Statistics to evaluate our reward function
        # TODO: Make numpy arrays but when saving convert back to python lists
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
        # Simulation methods related to routing
        self.routing_obj = Routing(beta=properties['beta'], topology=self.topology,
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
        self.epsilon -= amount
        if self.epsilon < 0.0:
            raise ValueError(f'Epsilon should be greater than 0 but it is {self.epsilon}')

    # TODO: Let's use numpy arrays and most likely need to update how average reward is calculated
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

    # TODO: Should be the same, double check
    def save_table(self):
        """
        Saves the current Q-table to a desired path.
        """
        create_dir(f'ai/models/q_tables/{self.table_path}')

        file_path = f'{os.getcwd()}/ai/models/q_tables/{self.table_path}/{self.sim_type}_table_c{self.cores_per_link}.npy'
        np.save(file_path, self.q_table)

        properties_dict = {
            'epsilon': self.epsilon,
            'episodes': self.episodes,
            'learn_rate': self.learn_rate,
            'discount_factor': self.discount,
            'reward_info': self.rewards_dict
        }
        with open(f'{os.getcwd()}/ai/models/q_tables/{self.table_path}/hyper_properties_c{self.cores_per_link}.json',
                  'w',
                  encoding='utf-8') as file:
            json.dump(properties_dict, file)

    # TODO: Should be the same, double check
    def load_table(self):
        """
        Loads a previously trained Q-table.
        """
        try:
            file_path = f'{os.getcwd()}/ai/models/q_tables/{self.table_path}/{self.sim_type}_table_c{self.cores_per_link}.npy'
            self.q_table = np.load(file_path)
        except FileNotFoundError:
            print('File not found, please ensure if you are testing, an already trained file exists and has been '
                  'specified correctly.')

        with open(f'{os.getcwd()}/ai/models/q_tables/{self.table_path}/hyper_properties_c{self.cores_per_link}.json',
                  encoding='utf-8') as file:
            properties_obj = json.load(file)
            self.epsilon = properties_obj['epsilon']
            self.learn_rate = properties_obj['learn_rate']
            self.discount = properties_obj['discount_factor']
            self.rewards_dict = properties_obj['reward_info']

    def _get_nli_cost(self):
        """
        Uses the routing object to get the non-linear impairment cost from the selected path.
        """
        mod_formats = self.mod_per_bw[self.chosen_bw]
        path_len = find_path_len(self.chosen_path, self.topology)
        mod_format = get_path_mod(mod_formats, path_len)

        self.routing_obj.net_spec_db = self.net_spec_db

        if not mod_format:
            return False

        self.routing_obj.slots_needed = self.mod_per_bw[self.chosen_bw][mod_format]['slots_needed']

        if self.nli_worst is None:
            self.nli_worst = self.routing_obj.find_worst_nli()
        self.nli_cost = self.routing_obj.nli_path(path=self.chosen_path)
        return self.nli_cost

    # TODO: Need to calculate intra-core XT worst case for different networks
    def _get_xt_cost(self):
        raise NotImplementedError

    def update_environment(self, routed: bool):
        """
        Updates the Q-learning environment.

        :param routed: Whether the path chosen was successfully routed or not.
        :type routed: bool
        """
        # TODO: Remove these to different policy methods
        if not routed:
            reward = -1.0
        else:
            # NLI worst relates to the worst NLI for a single link
            reward = 1.0 - (self.nli_cost / (self.nli_worst * float(len(self.chosen_path))))

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
                    self.learn_rate * (reward + (self.discount * max_future_q)))

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
                # TODO: Produce array of random numbers to attempt to find one at random, if you've checked all of them,
                #   - Then we must need to block
                next_node = np.random.randint(len(nodes))
                if str(next_node) in self.chosen_path or np.isnan(self.q_table[(curr_node, next_node)]):
                    continue
                self.chosen_path.append(str(next_node))
                found_next = True
            else:
                found_next, next_node = self._find_next_node(curr_node)

            if found_next is not False:
                if next_node == self.destination:
                    # TODO: Change for crosstalk
                    self._get_nli_cost()

                    return self.chosen_path

                curr_node = next_node
                nodes = self.q_table[next_node]
                continue

            # Q-routing chose too many nodes, no path found due to Q-routing constraint
            return False
