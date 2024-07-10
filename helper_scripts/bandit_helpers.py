import os
import json
import ast

import numpy as np
from sklearn.linear_model import LinearRegression

from helper_scripts.os_helpers import create_dir
from arg_scripts.rl_args import BanditProps


def load_model(train_fp: str):
    train_fp = os.path.join('logs', train_fp)
    with open(train_fp, 'r') as file_obj:
        state_vals_dict = json.load(file_obj)

    return state_vals_dict


def save_model(iteration: int, max_iters: int, len_rewards: int, num_requests: int, rewards_matrix: list,
               state_values_dict: dict, engine_props: dict, algorithm: str, is_path: bool):
    # TODO: Add save every 'x' iters to the configuration file
    if (iteration in (max_iters - 1, (max_iters - 1) % 50)) and len_rewards == num_requests:
        rewards_matrix = np.array(rewards_matrix)
        rewards_arr = rewards_matrix.mean(axis=0)

        date_time = os.path.join(engine_props['network'], engine_props['date'], engine_props['sim_start'])
        save_dir = os.path.join('logs', algorithm, date_time)
        create_dir(file_path=save_dir)

        erlang = engine_props['erlang']
        cores_per_link = engine_props['cores_per_link']
        if is_path:
            base_fp = f"e{erlang}_routes_c{cores_per_link}.npy"
        else:
            base_fp = f"e{erlang}_cores_c{cores_per_link}.npy"

        rewards_fp = f'rewards_{base_fp}'
        save_fp = os.path.join(os.getcwd(), save_dir, rewards_fp)
        np.save(save_fp, rewards_arr)

        if state_values_dict is None:
            return
        # Convert tuples to strings and arrays to lists for JSON format
        state_values_dict = {str(key): value.tolist() for key, value in state_values_dict.items()}

        if is_path:
            state_vals_fp = f"state_vals_e{erlang}_routes_c{cores_per_link}.json"
        else:
            state_vals_fp = f"state_vals_e{erlang}_cores_c{cores_per_link}.json"
        save_fp = os.path.join(os.getcwd(), save_dir, state_vals_fp)
        with open(save_fp, 'w') as file_obj:
            json.dump(state_values_dict, file_obj)


class EpsilonGreedyBandit:
    def __init__(self, rl_props: object, engine_props: dict, is_path: bool, is_core: bool):
        self.props = BanditProps()
        self.engine_props = engine_props
        self.rl_props = rl_props
        self.completed_sim = False
        self.iteration = 0
        self.is_path = is_path

        self.source = None
        self.dest = None
        self.path_index = None

        if is_path:
            self.n_arms = engine_props['k_paths']
        else:
            self.n_arms = engine_props['cores_per_link']

        self.epsilon = engine_props['epsilon_start']
        self.num_nodes = rl_props.num_nodes
        self.counts = {}
        self.values = {}

        # Initialize counts and values for each source-destination pair
        for source in range(self.num_nodes):
            for destination in range(self.num_nodes):
                if source == destination:
                    continue

                if is_path:
                    self.counts[(source, destination)] = np.zeros(self.n_arms)
                    self.values[(source, destination)] = np.zeros(self.n_arms)
                else:
                    for path_index in range(self.engine_props['k_paths']):
                        self.counts[(source, destination, path_index)] = np.zeros(self.n_arms)
                        self.values[(source, destination, path_index)] = np.zeros(self.n_arms)

    def select_path_arm(self, source: int, dest: int):
        self.source = source
        self.dest = dest
        pair = (source, dest)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            return np.argmax(self.values[pair])

    def select_core_arm(self, source: int, dest: int, path_index: int):
        self.source = source
        self.dest = dest
        self.path_index = path_index
        pair = (source, dest, path_index)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            return np.argmax(self.values[pair])

    def update(self, arm: int, reward: int, iteration: int):
        if self.is_path:
            pair = (self.source, self.dest)
        else:
            pair = (self.source, self.dest, self.path_index)

        self.counts[pair][arm] += 1
        n = self.counts[pair][arm]
        value = self.values[pair][arm]
        self.values[pair][arm] = value + (reward - value) / n

        if self.iteration >= len(self.props.rewards_matrix):
            self.props.rewards_matrix.append([])
        self.props.rewards_matrix[self.iteration].append(reward)

        # Check if we need to save the model
        save_model(iteration=iteration, max_iters=self.engine_props['max_iters'],
                   len_rewards=len(self.props.rewards_matrix[iteration]),
                   num_requests=self.engine_props['num_requests'],
                   rewards_matrix=self.props.rewards_matrix, engine_props=self.engine_props,
                   algorithm='greedy_bandit', is_path=self.is_path, state_values_dict=self.values)

    def setup_env(self):
        if not self.engine_props['is_training']:
            if self.is_path:
                self.values = load_model(train_fp=self.engine_props['path_model'])
                self.values = {ast.literal_eval(key): np.array(value) for key, value in self.values.items()}
            else:
                self.values = load_model(train_fp=self.engine_props['core_model'])
                self.values = {ast.literal_eval(key): np.array(value) for key, value in self.values.items()}


class UCBBandit:
    def __init__(self, rl_props: object, engine_props: dict, is_core: bool, is_path: bool):
        # TODO: Discrepancy here, why don't we use a class props?
        self.props = {'rewards_matrix': []}
        self.engine_props = engine_props
        self.rl_props = rl_props
        self.completed_sim = False
        self.iteration = 0
        self.is_path = is_path

        self.path_index = None
        self.source = None
        self.dest = None

        self.num_nodes = rl_props.num_nodes
        self.counts = {}
        self.values = {}

        if is_path:
            self.n_arms = engine_props['k_paths']
        else:
            self.n_arms = engine_props['cores_per_link']

        # Initialize counts and values for each source-destination pair
        for source in range(self.num_nodes):
            for destination in range(self.num_nodes):
                if source == destination:
                    continue
                if is_path:
                    self.counts[(source, destination)] = np.zeros(self.n_arms)
                    self.values[(source, destination)] = np.zeros(self.n_arms)
                else:
                    for path_index in range(self.engine_props['k_paths']):
                        self.counts[(source, destination, path_index)] = np.zeros(self.n_arms)
                        self.values[(source, destination, path_index)] = np.zeros(self.n_arms)

    # TODO: Debug these functions
    def select_path_arm(self, source: int, dest: int):
        self.source = source
        self.dest = dest
        pair = (source, dest)
        if 0 in self.counts[pair]:
            return np.argmin(self.counts[pair])
        else:
            total_counts = sum(self.counts[pair])
            ucb_values = self.values[pair] + np.sqrt(2 * np.log(total_counts) / self.counts[pair])
            return np.argmax(ucb_values)

    def select_core_arm(self, source: int, dest: int, path_index: int):
        self.source = source
        self.dest = dest
        self.path_index = path_index
        pair = (source, dest, path_index)

        if 0 in self.counts[pair]:
            return np.argmin(self.counts[pair])  # Ensure each arm is tried at least once
        else:
            total_counts = sum(self.counts[pair])
            ucb_values = self.values[pair] + np.sqrt(2 * np.log(total_counts) / self.counts[pair])
            return np.argmax(ucb_values)

    def update(self, arm: int, reward: int, iteration: int):
        if self.is_path:
            pair = (self.source, self.dest)
        else:
            pair = (self.source, self.dest, self.path_index)

        self.counts[pair][arm] += 1
        n = self.counts[pair][arm]
        value = self.values[pair][arm]
        self.values[pair][arm] = value + (reward - value) / n

        if self.iteration >= len(self.props['rewards_matrix']):
            self.props['rewards_matrix'].append([])
        self.props['rewards_matrix'][self.iteration].append(reward)

        # Check if we need to save the model
        save_model(iteration=iteration, max_iters=self.engine_props['max_iters'],
                   len_rewards=len(self.props['rewards_matrix'][iteration]),
                   num_requests=self.engine_props['num_requests'],
                   rewards_matrix=self.props['rewards_matrix'], engine_props=self.engine_props,
                   algorithm='ucb_bandit', is_path=self.is_path, state_values_dict=self.values)

    def setup_env(self):
        if not self.engine_props['is_training']:
            if self.is_path:
                self.values = load_model(train_fp=self.engine_props['path_model'])
                self.values = {ast.literal_eval(key): np.array(value) for key, value in self.values.items()}
            else:
                self.values = load_model(train_fp=self.engine_props['core_model'])
                self.values = {ast.literal_eval(key): np.array(value) for key, value in self.values.items()}


class ThompsonSamplingBandit:
    def __init__(self, rl_props: dict, engine_props: dict, is_path: bool, is_core: bool):
        # TODO: Why don't we use bandit props?
        self.props = {'rewards_matrix': []}
        self.engine_props = engine_props
        self.rl_props = rl_props
        self.completed_sim = False
        self.iteration = 0

        self.path_index = None
        self.source = None
        self.dest = None

        self.num_nodes = rl_props['num_nodes']
        self.successes = {}
        self.failures = {}

        self.is_path = is_path
        if is_path:
            self.n_arms = engine_props['k_paths']
        else:
            self.n_arms = engine_props['cores_per_link']

        # Initialize successes and failures for each source-destination pair
        for source in range(self.num_nodes):
            for destination in range(self.num_nodes):
                if source == destination:
                    continue
                if is_path:
                    self.successes[(source, destination)] = np.ones(self.n_arms)
                    self.failures[(source, destination)] = np.ones(self.n_arms)
                else:
                    for path_index in range(self.engine_props['k_paths']):
                        self.successes[(source, destination, path_index)] = np.ones(self.n_arms)
                        self.failures[(source, destination, path_index)] = np.ones(self.n_arms)

    # TODO: Debug these functions (can probably use one function for all)
    def select_path_arm(self, source: int, dest: int):
        self.source = source
        self.dest = dest
        pair = (source, dest)

        samples = np.random.beta(self.successes[pair], self.failures[pair])
        return np.argmax(samples)

    def select_core_arm(self, source: int, dest: int, path_index: int):
        self.source = source
        self.dest = dest
        self.path_index = path_index
        pair = (source, dest, path_index)

        samples = np.random.beta(self.successes[pair], self.failures[pair])
        return np.argmax(samples)

    def update(self, arm: int, reward: int, iteration: int):
        if self.is_path:
            pair = (self.source, self.dest)
        else:
            pair = (self.source, self.dest, self.path_index)

        if reward > 0:
            self.successes[pair][arm] += 1
        else:
            self.failures[pair][arm] += 1

        if self.iteration >= len(self.props['rewards_matrix']):
            self.props['rewards_matrix'].append([])
        self.props['rewards_matrix'][self.iteration].append(reward)

        # Check if we need to save the model
        save_model(iteration=iteration, max_iters=self.engine_props['max_iters'],
                   len_rewards=len(self.props['rewards_matrix'][iteration]),
                   num_requests=self.engine_props['num_requests'],
                   rewards_matrix=self.props['rewards_matrix'], engine_props=self.engine_props,
                   algorithm='thompson_sampling_bandit', is_path=self.is_path, state_values_dict=None)

    # TODO: Complete
    def setup_env(self):
        pass


# TODO: Class no longer working
class ContextualEpsilonGreedyBandit:
    def __init__(self, rl_props: object, engine_props: dict, is_path: bool, is_core: bool):
        self.props = {'rewards_matrix': []}
        self.is_path = is_path
        self.engine_props = engine_props
        if is_path:
            self.n_arms = engine_props['k_paths']
        else:
            self.n_arms = engine_props['cores_per_link']

        self.epsilon = engine_props['epsilon_start']
        self.rl_props = rl_props

        n_features = (engine_props['topology'].number_of_nodes() * 2) + self.n_arms

        self.num_nodes = rl_props['num_nodes']
        self.models = {}
        self.X = {}
        self.y = {}
        self.is_fitted = {}

        self.path_index = None
        self.source = None
        self.dest = None

        # Initialize models, X, y, and is_fitted for each source-destination pair
        for source in range(self.num_nodes):
            for destination in range(self.num_nodes):
                if source == destination:
                    continue

                if is_path:
                    pairs_list = [(source, destination)]
                else:
                    pairs_list = list()
                    for path_index in range(self.engine_props['k_paths']):
                        pairs_list.append((source, destination, path_index))

                for pair in pairs_list:
                    self.models[pair] = [LinearRegression() for _ in range(self.n_arms)]
                    self.X[pair] = [np.empty((0, n_features)) for _ in range(self.n_arms)]
                    self.y[pair] = [np.empty((0,)) for _ in range(self.n_arms)]
                    self.is_fitted[pair] = [False] * self.n_arms

    # TODO: Debug these methods
    def select_path_arm(self, source: int, dest: int, context):
        self.source = source
        self.dest = dest
        pair = (source, dest)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            estimated_rewards = []
            for i, model in enumerate(self.models[pair]):
                if self.is_fitted[pair][i]:
                    estimated_reward = model.predict(context.reshape(1, -1))[0]
                else:
                    estimated_reward = 0
                estimated_rewards.append(estimated_reward)
            return np.argmax(estimated_rewards)

    def select_core_arm(self, source: int, dest: int, context, path_index):
        self.source = source
        self.dest = dest
        self.path_index = path_index
        pair = (source, dest, path_index)

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            estimated_rewards = []
            for i, model in enumerate(self.models[pair]):
                if self.is_fitted[pair][i]:
                    estimated_reward = model.predict(context.reshape(1, -1))[0]
                else:
                    estimated_reward = 0
                estimated_rewards.append(estimated_reward)
            return np.argmax(estimated_rewards)

    def update(self, arm: int, context, reward: float, iteration: int):
        if self.is_path:
            pair = (self.source, self.dest)
        else:
            pair = (self.source, self.dest, self.path_index)

        self.X[pair][arm] = np.vstack([self.X[pair][arm], context])
        self.y[pair][arm] = np.append(self.y[pair][arm], reward)

        self.models[pair][arm].fit(self.X[pair][arm], self.y[pair][arm])
        self.is_fitted[pair][arm] = True

        if iteration >= len(self.props['rewards_matrix']):
            self.props['rewards_matrix'].append([])
        self.props['rewards_matrix'][iteration].append(reward)

        # Check if we need to save the model
        save_model(iteration=iteration, max_iters=self.engine_props['max_iters'],
                   len_rewards=len(self.props['rewards_matrix'][iteration]),
                   num_requests=self.engine_props['num_requests'],
                   rewards_matrix=self.props['rewards_matrix'], engine_props=self.engine_props,
                   algorithm='thompson_sampling_bandit', is_path=self.is_path, state_values_dict=self.values)

    def setup_env(self):
        raise NotImplementedError


# TODO: Need to change context, no longer supported
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
