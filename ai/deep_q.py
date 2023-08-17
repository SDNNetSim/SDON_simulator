import numpy as np
import gymnasium as gym
from stable_baselines3 import A2C


# TODO: Check env


class DeepQ:

    def __init__(self):
        pass

    @staticmethod
    def set_seed(seed: int):
        """
        Used to set the seed for controlling 'random' generation.

        :param seed: The seed to be set for numpy random generation.
        :type seed: int
        """
        np.random.seed(seed)

    def _update_rewards_dict(self):
        raise NotImplementedError

    @staticmethod
    def update_environment(model, obs, vec_env):
        for i in range(1000):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            vec_env.render('human')

    def setup_environment(self):
        env = gym.make('CartPole-v1')
        model = A2C('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=10_000)

        vec_env = model.get_env()
        obs = vec_env.reset()

        self.update_environment(model, obs, vec_env)


if __name__ == '__main__':
    test_obj = DeepQ()
    test_obj.setup_environment()
