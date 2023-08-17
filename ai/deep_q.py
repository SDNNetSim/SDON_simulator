# Standard library imports
import os
import json

# Third party imports
import numpy as np
import networkx as nx

# Local application imports
from useful_functions.handle_dirs_files import create_dir


class QLearning:
    """
    Controls methods related to the Q-learning reinforcement learning algorithm.
    """

    def __init__(self):
        raise NotImplementedError

    @staticmethod
    def set_seed(seed: int):
        """
        Used to set the seed for controlling 'random' generation.

        :param seed: The seed to be set for numpy random generation.
        :type seed: int
        """
        np.random.seed(seed)

    def decay_epsilon(self):
        raise NotImplementedError

    def _update_rewards_dict(self):
        raise NotImplementedError

    def update_environment(self):
        raise NotImplementedError

    def setup_environment(self):
        raise NotImplementedError

    def route(self):
        raise NotImplementedError


if __name__ == '__main__':
    pass
