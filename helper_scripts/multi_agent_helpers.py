class PathAgent:
    def __init__(self):
        raise NotImplementedError

    def get_obs(self):
        raise NotImplementedError

    def get_action(self):
        raise NotImplementedError

    def get_reward(self):
        raise NotImplementedError

    def get_route(self):
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

    def get_core(self):
        raise NotImplementedError


class SpectrumAgent:
    def __init__(self):
        raise NotImplementedError

    def get_obs(self):
        raise NotImplementedError

    def get_action(self):
        raise NotImplementedError

    def get_reward(self):
        raise NotImplementedError

    def get_spectrum(self):
        raise NotImplementedError
