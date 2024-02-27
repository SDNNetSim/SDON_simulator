import gymnasium as gym
from gymnasium import spaces
import numpy as np


# TODO: I've decided on making this script very similar to engine, it seems to be the easiest way
# TODO: Probably need to include render and close as methods as well...


class DQNEnv(gym.Env):
    metadata = None

    def __init__(self):
        super().__init__()
        self.observation_space = None
        self.action_space = None

    def step(self, action):
        success, new_state = self.simulator.allocate_request(action)
        reward = self._calculate_reward(success)

        terminated = self._check_terminated(new_state)
        truncated = False
        info = {}

        return new_state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.simulator.reset()
        return self.simulator.get_network_state(), {}

    @staticmethod
    def _calculate_reward(success):
        return 1 if success else -1

    def terminate(self, observation):
        return self.simulator.is_simulation_complete()

    def render(self):
        pass

    def close(self):
        pass
