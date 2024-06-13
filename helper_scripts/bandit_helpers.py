import numpy as np

from arg_scripts.rl_args import empty_bandit_props


# TODO: Only epsilon greedy for now but will add another one
class MultiBanditHelpers:
    def __init__(self, rl_props: dict, engine_props: dict):
        self.props = empty_bandit_props
        self.engine_props = engine_props
        self.rl_props = rl_props
        self.completed_sim = False
        self.iteration = 0

        # TODO: Change off of engine props
        self.n_arms = engine_props['k_paths']
        self.epsilon = engine_props['epsilon_start']
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)

    def select_arm(self):
        if np.random.rand() < self.epsilon:
            # Exploration: choose a random arm
            return np.random.randint(self.n_arms)
        else:
            # Exploitation: choose the best-known arm
            return np.argmax(self.values)

    def update(self, arm, reward):
        # Update counts and values (using incremental formula)
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        # Incremental update of the mean
        self.values[arm] = value + (reward - value) / n
