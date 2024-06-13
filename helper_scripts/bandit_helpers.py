import numpy as np

from sklearn.linear_model import LinearRegression

from arg_scripts.rl_args import empty_bandit_props


# TODO: Save model and other results (Use this as a baseline?)
#   - For a contextual bandit, similar to the q-table:
#       - Source, destination, each of the k-paths
#       - Later on, congestion levels?
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


class ContextualGreedyHelpers:
    def __init__(self, rl_props: dict, engine_props: dict):
        self.n_arms = engine_props['k_paths']
        self.epsilon = engine_props['epsilon_start']
        self.rl_props = rl_props

        # To consider the source, destination, and congestion of all K-paths
        n_features = 2 + engine_props['k_paths']
        self.models = [LinearRegression() for _ in range(self.n_arms)]
        self.X = [np.empty((0, n_features)) for _ in range(self.n_arms)]
        self.y = [np.empty((0,)) for _ in range(self.n_arms)]
        self.is_fitted = [False] * self.n_arms

    def select_arm(self, context):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            estimated_rewards = []
            for i, model in enumerate(self.models):
                if self.is_fitted[i]:
                    estimated_reward = model.predict(context.reshape(1, -1))[0]
                else:
                    estimated_reward = 0
                estimated_rewards.append(estimated_reward)
            return np.argmax(estimated_rewards)

    def update(self, arm, context, reward):
        self.X[arm] = np.vstack([self.X[arm], context])
        self.y[arm] = np.append(self.y[arm], reward)

        self.models[arm].fit(self.X[arm], self.y[arm])
        self.is_fitted[arm] = True

    def setup_env(self):
        pass


class ContextGenerator:
    def __init__(self, rl_props: dict, engine_props: dict):
        self.num_sources = engine_props['topology'].number_of_nodes()
        self.num_destinations = engine_props['topology'].number_of_nodes()
        self.num_paths = engine_props['k_paths']

        self.curr_context = None

    def generate_context(self, source, dest, congestion_levels):
        source_vec = np.zeros(self.num_sources)
        source_vec[source] = 1

        dest_vec = np.zeros(self.num_destinations)
        dest_vec[dest] = 1
        self.curr_context = np.concatenate([source_vec, dest_vec, congestion_levels])
