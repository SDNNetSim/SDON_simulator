import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CustomEnv(gym.Env):
    # TODO: Change
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()

        discrete_actions = 10
        self.action_space = spaces.Discrete(discrete_actions)

        channels = 5
        height = 1
        width = 20
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(channels, height, width), dtype=np.uint8)

    def step(self, action):
        observation = None
        reward = None
        terminated = None
        truncated = None
        info = None

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        observation = None
        info = None

        return observation, info

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
