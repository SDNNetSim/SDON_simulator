class PathAgent:
    def __init__(self):
        raise NotImplementedError

    def get_obs(self):
        raise NotImplementedError

    def get_action(self):
        raise NotImplementedError

    def get_reward(self):
        raise NotImplementedError


class CoreAgent:
    def __init__(self):
        raise NotImplementedError

    def get_obs(self):
        raise NotImplementedError

    def get_action(self):
        raise NotImplementedError

    def get_reward(self):
        raise NotImplementedError


# TODO: Can be PPO, DQN, or A2C
class SpectrumAgent:
    def __init__(self):
        raise NotImplementedError

    def get_obs(self):
        raise NotImplementedError

    def get_action(self):
        raise NotImplementedError

    def get_reward(self):
        raise NotImplementedError
