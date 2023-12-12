import numpy as np
import gymnasium as gym
from stable_baselines3 import A2C


# TODO: Check env


class DeepQ:
    """
    Update
    """

    def __init__(self):
        pass

    @staticmethod
    def set_seed(seed: int):
        """
        Used to set the seed for controlling 'random' generation.

        :param seed: The seed to be set for numpy random generation.
        :type seed: int

        :return: None
        """
        np.random.seed(seed)

    def _update_rewards_dict(self):
        """
        Raise not implemented error (exception) on call

        :return: None
        """
        raise NotImplementedError

    @staticmethod
    def update_environment(model, obs, vec_env):
        """
        Update the reinforcement learning environment using the provided A2C model, observation, and vectorized environment.

        :param model: The A2C model used for reinforcement learning.
        :type model: Any
        :param obs: The initial observation state.
        :type obs: Any
        :param vec_env: The vectorized environment for training the model.
        :type vec_env: Any

        :return: None
        """
        for _ in range(1000):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            vec_env.render('human')

            print(reward, done, info)

    def setup_environment(self):
        """
        Set up the reinforcement learning environment using the OpenAI Gym's CartPole environment and A2C model.

        :return: None
        """
        env = gym.make('CartPole-v1')
        model = A2C('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=10_000)

        vec_env = model.get_env()
        obs = vec_env.reset()

        self.update_environment(model, obs, vec_env)


if __name__ == '__main__':
    test_obj = DeepQ()
    test_obj.setup_environment()
