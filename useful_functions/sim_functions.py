# Standard library imports
from typing import List

# Third-party library imports
import networkx as nx

# Local application imports
from ai.reinforcement_learning import *


def get_path_mod(mod_formats: dict, path_len: int):
    """
    Given an object of modulation formats and maximum lengths, choose the one that satisfies the requirements.

    :param mod_formats: The modulation object, holds needed information for maximum reach
    :type mod_formats: dict
    :param path_len: The length of the path to be taken
    :type path_len: int

    :return: The chosen modulation format, or false
    """
    if mod_formats['QPSK']['max_length'] >= path_len > mod_formats['16-QAM']['max_length']:
        resp = 'QPSK'
    elif mod_formats['16-QAM']['max_length'] >= path_len > mod_formats['64-QAM']['max_length']:
        resp = '16-QAM'
    elif mod_formats['64-QAM']['max_length'] >= path_len:
        resp = '64-QAM'
    else:
        return False

    return resp


def sort_dict_keys(dictionary: dict):
    """
    Given a dictionary with key-value pairs, return a new dictionary with the same pairs, sorted by keys in descending
    order.

    :param dictionary: The dictionary to sort.
    :type dictionary: dict

    :return: A new dictionary with the same pairs as the input dictionary, but sorted by keys in descending order.
    :rtype: dict
    """
    sorted_keys = sorted(map(int, dictionary.keys()), reverse=True)
    sorted_dict = {str(key): dictionary[str(key)] for key in sorted_keys}

    return sorted_dict


def find_path_len(path: List[int], topology: nx.Graph):
    """
    Finds the length of a path in a physical topology.

    :param path: A list of integers representing the nodes in the path.
    :type path: list of int
    :param topology: A networkx graph object representing the physical topology of the simulation.
    :type topology: networkx.Graph

    :return: The length of the path.
    """
    path_len = 0
    for i in range(len(path) - 1):
        path_len += topology[path[i]][path[i + 1]]['length']

    return path_len


class AIMethods:
    """
    Contains methods related to artificial intelligence for the simulator.
    """

    def __init__(self, algorithm: str = None, sim_info: str = None, max_segments: int = None, is_training: bool = None,
                 topology: nx.Graph = None, seed: int = None):
        """
        Initializes the AIMethods class.

        :param algorithm: The type of AI algorithm to be used in the simulation.
        :type algorithm: str

        :param sim_info: Relevant information to the current running simulation.
        :type sim_info: str

        :param max_segments: The maximum allowed segments allowed for a single request.
        :type max_segments: int

        :param is_training: Determines if the simulation is training or testing.
        :type is_training: bool

        :param topology: The network topology.
        :type topology: nx.Graph

        :param seed: The seed used for random generation.
        :type seed: int
        """
        self.algorithm = algorithm
        self.sim_info = sim_info
        self.max_segments = max_segments
        self.is_training = is_training
        self.topology = topology
        self.seed = seed

        # An object for the chosen AI algorithm
        self.ai_obj = None

    def _q_save(self):
        """
        Saves the current state of the Q-table.
        """
        self.ai_obj.save_table(path=self.sim_info, max_segments=self.max_segments)

    def _q_update_env(self, routed: bool, path: list, free_slots: int, iteration: int):
        """
        Updates the Q-learning environment.

        :param routed: A flag to determine if a request was routed or not.
        :type routed: bool

        :param path: The path for the request.
        :type path: list

        :param free_slots: The number of total free slots in the path.
        :type free_slots: int

        :param iteration: The current iteration of the simulation.
        :type iteration: int
        """
        self.ai_obj.update_environment(routed=routed, path=path, free_slots=free_slots)

        # Decay epsilon for half of the iterations evenly each time
        if 1 <= iteration <= iteration // 2 and self.is_training:
            decay_amount = (self.ai_obj.epsilon / (iteration // 2) - 1)
            self.ai_obj.decay_epsilon(amount=decay_amount)

    def _q_spectrum(self):
        raise NotImplementedError

    def _q_routing(self, source: int, destination: int, mod_formats: dict):
        """
        Given a request, determines the path from source to destination.

        :param source: The source node.
        :type source: int

        :param destination: The destination node.
        :type destination: int

        :param mod_formats: Modulation format info related to the size of the request (in Gbps).
        :type mod_formats: dict

        :return: A path from source to destination and a modulation format.
        :rtype: list, str
        """
        path, path_mod = self.ai_obj.route(source=source, destination=destination, mod_formats=mod_formats)
        return path, path_mod

    def _init_q_learning(self, epsilon: float = 0.2, episodes: int = 1000, learn_rate: float = 0.5,
                         discount: float = 0.2, trained_table: str = None):
        """
        Initializes a QLearning class and sets up the initial environment and Q-table.

        :param epsilon: Parameter in the bellman equation to determine degree of randomness.
        :type epsilon: float

        :param episodes: The number of iterations or simulations.
        :type episodes: int

        :param learn_rate: The learning rate, alpha, in the bellman equation.
        :type learn_rate: float

        :param discount: The discount factor in the bellman equation to determine the balance between current and
                         future rewards.
        :type discount: float

        :param trained_table: A path to an already trained Q-table.
        :type trained_table: str
        """
        self.ai_obj = QLearning(epsilon=epsilon, episodes=episodes, learn_rate=learn_rate, discount=discount,
                                topology=self.topology)
        self.ai_obj.setup_environment()

        # Load a pre-trained table or start a new one
        if self.is_training:
            self.ai_obj.load_table(path=self.sim_info, max_segments=self.max_segments)
        else:
            self.ai_obj.load_table(path=trained_table, max_segments=self.max_segments)

        self.ai_obj.set_seed(self.seed)

    def save(self):
        """
        Responsible for saving relevant information.
        """
        if self.algorithm == 'q_learning':
            self._q_save()

    def route(self, **kwargs):
        """
        Responsible for routing.
        """
        if self.algorithm == 'q_learning':
            self._q_routing()

    def update(self, **kwargs):
        """
        Responsible for updating environment information.
        """
        if self.algorithm == 'q_learning':
            self._q_update_env(routed=kwargs['routed'], path=kwargs['path'], free_slots=kwargs['free_slots'],
                               iteration=kwargs['iteration'])

    def setup(self, **kwargs):
        """
        Responsible for setting up available AI algorithms and their methods.
        """
        if self.algorithm == 'q_learning':
            # TODO: Update
            self._init_q_learning()
        else:
            raise NotImplementedError(f'Algorithm: {self.algorithm} not recognized.')
