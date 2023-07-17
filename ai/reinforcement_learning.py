# Third party imports
import numpy as np
import networkx as nx

# Local application imports
from useful_functions.sim_functions import get_path_mod, find_path_len


# TODO: Topology is hard coded in some methods


class QLearning:
    """
    Controls methods related to the Q-learning reinforcement learning algorithm.
    """

    def __init__(self, epsilon: float = 0.1, epsilon_decay: float = 0.01, episodes: int = 1000, learn_rate: float = 0.1,
                 discount: float = 0.95, topology: nx.Graph = None):
        """
        Initializes the QLearning class.

        :param epsilon: Parameter in the bellman equation to determine degree of randomness.
        :type epsilon: float

        :param epsilon_decay: Determines the decay in the degree of randomness for every episode.
        :type epsilon_decay: float

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
        # TODO: Use epsilon decay
        self.epsilon_decay = epsilon_decay
        self.episodes = episodes
        self.learn_rate = learn_rate
        self.discount = discount
        self.topology = topology

        # Statistics used for plotting
        self.rewards_dict = {'episode': [], 'average': [], 'min': [], 'max': [], 'rewards': []}

    @staticmethod
    def set_seed(seed: int = None):
        """
        Used to set the seed for controlling 'random' generation.

        :param seed: The seed to be set for numpy random generation.
        :type seed: int
        """
        np.random.seed(seed)

    def plot_rewards(self):
        """
        Plots reward values vs. episodes.
        """
        raise NotImplementedError

    # TODO: Add better reward scheme (based on congestion, distance, mod format, SNR, etc.)
    #   Potentially based on time for example, congestion at first and distance later (different reward weights)
    def update_environment(self, routed, path):
        """
        The custom environment that updates the Q-table with respect to a reward policy.

        :param routed: If a request was successfully routed or not.
        :type routed: bool

        :param path: The path taken from source to destination.
        :type path: list

        :return: The reward value.
        :rtype: int
        """
        if routed:
            reward = 1.0
        else:
            reward = -1.0

        for i in range(len(path) - 1):
            state = path[i]
            new_state = path[i + 1]

            # If terminating node, only consider the reward (there is no future state)
            if i + 1 == len(path):
                max_future_q = reward
            else:
                max_future_q = np.max(self.q_table[new_state])

            current_q = self.q_table[(state, new_state)]
            new_q = ((1.0 - self.learn_rate) * current_q) + (
                    self.learn_rate * (reward + self.discount * max_future_q))

            self.q_table[(state, new_state)] = new_q

    def setup_environment(self):
        """
        Initializes the environment.
        """
        # Init q-table for USNet, a 24 node network
        self.q_table = np.zeros(24, 24)

        for source in range(0, 24):
            for destination in range(0, 24):
                if source == destination:
                    self.q_table[(source, destination)] = np.nan
                    continue

                # A link exists between these nodes, init to zero
                if source in self.topology.neighbors(destination):
                    self.q_table[(source, destination)] = 0
                else:
                    self.q_table[(source, destination)] = np.nan

    def route(self, source, destination, mod_formats):
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
        path = [source]
        while True:
            random_float = np.round(np.random.uniform(0, 1), decimals=1)
            while True:
                # Choose a random action with respect to epsilon
                if random_float < self.epsilon:
                    next_node = np.random.randint(24)
                else:
                    next_node = np.argmax(self.q_table[source])

                q_value = self.q_table[(int(source), next_node)]
                # No connection exists between these nodes
                if np.isnan(q_value):
                    continue

                path.append(next_node)
                if next_node == destination:
                    path_len = find_path_len(path, self.topology)
                    mod_format = get_path_mod(mod_formats, path_len)
                    return path, mod_format


if __name__ == '__main__':
    pass
