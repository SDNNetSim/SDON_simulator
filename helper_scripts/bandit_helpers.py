import numpy as np

from arg_scripts.rl_args import empty_bandit_props


class MultiBanditHelpers:
    def __init__(self, rl_props: dict, engine_props: dict):
        self.props = empty_bandit_props
        self.engine_props = engine_props
        self.rl_props = rl_props
        self.completed_sim = False
        self.iteration = 0

        self.n_arms = engine_props['k_paths']
        self.epsilon = engine_props['epsilon_start']
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)

    def select_arm(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            return np.argmax(self.values)

    def update(self, arm: int, reward: int):
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = value + (reward - value) / n

        try:
            self.props['rewards_matrix'][self.iteration].append(reward)
        except IndexError:
            self.props['rewards_matrix'].append([reward])

    def setup_env(self):
        pass
