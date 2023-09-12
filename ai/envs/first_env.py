import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CustomEnv(gym.Env):
    """
    Update
    """
    LEFT = 0
    RIGHT = 1

    def __init__(self):
        # TODO: What?
        super().__init__()

        discrete_actions = 2
        self.agent_value = 0
        self.action_space = spaces.Discrete(discrete_actions)

        self.observation_space = spaces.Box(low=0, high=100, shape=(1,), dtype=np.uint8)

    def step(self, action):
        if action == self.LEFT:
            self.agent_value += 1
            reward = 1
        elif action == self.RIGHT:
            self.agent_value -= 1
            reward = 0
        else:
            raise ValueError('Received an invalid action.')

        if self.agent_value == 100:  # pylint: disable=simplifiable-if-statement
            terminated = True
        else:
            terminated = False

        observation = self.agent_value
        # Unused variables for now but suggested to have per the documentation
        truncated = None
        info = None

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):  # pylint: disable=arguments-differ
        self.agent_value = 0

        observation = self.agent_value
        info = None

        return observation, info

    # TODO: Probably not needed these methods since we won't use a GUI
    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
