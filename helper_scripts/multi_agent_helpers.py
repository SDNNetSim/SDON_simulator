import os

import numpy as np
from gymnasium import spaces

from .ql_helpers import QLearningHelpers
from .bandit_helpers import EpsilonGreedyBandit
from .bandit_helpers import UCBBandit


class PathAgent:
    """
    A class that handles everything related to path assignment in reinforcement learning simulations.
    """

    def __init__(self, path_algorithm: str, rl_props: object, rl_help_obj: object):
        self.path_algorithm = path_algorithm
        self.engine_props = None
        self.rl_props = rl_props
        self.rl_help_obj = rl_help_obj
        self.agent_obj = None
        self.context_obj = None

        self.level_index = None
        self.cong_list = None

    def end_iter(self):
        """
        Ends an iteration for the path agent.
        """
        if self.path_algorithm == 'q_learning':
            self.agent_obj.decay_epsilon()

    def setup_env(self):
        """
        Sets up the environment for the path agent.
        """
        if self.path_algorithm == 'q_learning':
            self.agent_obj = QLearningHelpers(rl_props=self.rl_props, engine_props=self.engine_props)
            self.agent_obj.setup_env()
        elif self.path_algorithm == 'epsilon_greedy_bandit':
            self.agent_obj = EpsilonGreedyBandit(rl_props=self.rl_props, engine_props=self.engine_props, is_path=True)
        elif self.path_algorithm == 'ucb_bandit':
            self.agent_obj = UCBBandit(rl_props=self.rl_props, engine_props=self.engine_props, is_path=True)
        else:
            raise NotImplementedError

    def get_reward(self, was_allocated: bool, path_length: int):
        """
        Get the current reward for the last agent's action.

        :param was_allocated: If the request was allocated or not.
        :param path_length: The path length.
        :return: The reward.
        :rtype: float
        """
        if was_allocated:
            return self.engine_props['reward']

        return self.engine_props['penalty'] - path_length

    def update(self, was_allocated: bool, net_spec_dict: dict, iteration: int, path_length: int):
        """
        Makes updates to the agent for each time step.

        :param was_allocated: If the request was allocated.
        :param net_spec_dict: The current network spectrum database.
        :param path_length: Length of the path.
        :param iteration: The current iteration.
        """
        reward = self.get_reward(was_allocated=was_allocated, path_length=path_length)

        self.agent_obj.iteration = iteration
        if self.path_algorithm == 'q_learning':
            self.agent_obj.update_routes_matrix(reward=reward, level_index=self.level_index,
                                                net_spec_dict=net_spec_dict)
        elif self.path_algorithm == 'epsilon_greedy_bandit':
            self.agent_obj.update(reward=reward, arm=self.rl_props.chosen_path_index, iteration=iteration)
        elif self.path_algorithm == 'ucb_bandit':
            self.agent_obj.update(reward=reward, arm=self.rl_props.chosen_path_index, iteration=iteration)
        else:
            raise NotImplementedError

    def __ql_route(self, random_float: float):
        if random_float < self.agent_obj.props.epsilon:
            self.rl_props.chosen_path_index = np.random.choice(self.rl_props.k_paths)
            # The level will always be the last index
            self.level_index = self.cong_list[self.rl_props.chosen_path_index][-1]

            if self.rl_props.chosen_path_index == 1 and self.rl_props.k_paths == 1:
                self.rl_props.chosen_path_index = 0
            self.rl_props.chosen_path_list = self.rl_props.paths_list[self.rl_props.chosen_path_index]
        else:
            self.rl_props.chosen_path_index, self.rl_props.chosen_path_list = self.agent_obj.get_max_curr_q(
                cong_list=self.cong_list, matrix_flag='routes_matrix')
            self.level_index = self.cong_list[self.rl_props.chosen_path_index][-1]

    def _ql_route(self):
        random_float = float(np.round(np.random.uniform(0, 1), decimals=1))
        routes_matrix = self.agent_obj.props.routes_matrix
        self.rl_props.paths_list = routes_matrix[self.rl_props.source][self.rl_props.destination]['path']

        self.cong_list = self.rl_help_obj.classify_paths(paths_list=self.rl_props.paths_list)
        if self.rl_props.paths_list.ndim != 1:
            self.rl_props.paths_list = self.rl_props.paths_list[:, 0]

        self.__ql_route(random_float=random_float)

        if len(self.rl_props.chosen_path_list) == 0:
            raise ValueError('The chosen path can not be None')

    # TODO: Ideally q-learning should be like this (agent_obj.something)
    def _bandit_route(self, route_obj: object):
        paths_list = route_obj.route_props.paths_matrix
        source = paths_list[0][0]
        dest = paths_list[0][-1]
        self.rl_props.chosen_path_index = self.agent_obj.select_path_arm(source=int(source), dest=int(dest))
        self.rl_props.chosen_path_list = route_obj.route_props.paths_matrix[self.rl_props.chosen_path_index]

    def get_route(self, **kwargs):
        """
        Assign a route for the current request.
        """
        if self.path_algorithm == 'q_learning':
            self._ql_route()
        elif self.path_algorithm in ('epsilon_greedy_bandit', 'thompson_sampling_bandit', 'ucb_bandit'):
            self._bandit_route(route_obj=kwargs['route_obj'])
        else:
            raise NotImplementedError

    def load_model(self, model_path: str, erlang: float, num_cores: int):
        """
        Loads a previously trained path agent model.

        :param model_path: The path to the trained model.
        :param erlang: The Erlang value the model was trained with.
        :param num_cores: The number of cores the model was trained with.
        """
        self.setup_env()
        if self.engine_props['path_algorithm'] == 'q_learning':
            model_path = os.path.join('logs', model_path, f'e{erlang}_routes_c{num_cores}.npy')
            self.agent_obj.props.routes_matrix = np.load(model_path, allow_pickle=True)


class CoreAgent:
    """
    A class that handles everything related to core assignment in reinforcement learning simulations.
    """

    def __init__(self, core_algorithm: str, rl_props: object, rl_help_obj: object):
        self.core_algorithm = core_algorithm
        self.rl_props = rl_props
        self.engine_props = None
        self.agent_obj = None
        self.rl_help_obj = rl_help_obj

        self.level_index = None
        self.cong_list = list()
        self.no_penalty = False
        self.ramp_up = False

    def end_iter(self):
        """
        Ends an iteration for the core agent.
        """
        if self.core_algorithm == 'q_learning':
            self.agent_obj.decay_epsilon()

    def setup_env(self):
        """
        Sets up the environment for the core agent.
        """
        if self.core_algorithm == 'q_learning':
            self.agent_obj = QLearningHelpers(rl_props=self.rl_props, engine_props=self.engine_props)
        elif self.core_algorithm == 'epsilon_greedy_bandit':
            self.agent_obj = EpsilonGreedyBandit(rl_props=self.rl_props, engine_props=self.engine_props, is_path=False)
        elif self.core_algorithm == 'ucb_bandit':
            self.agent_obj = UCBBandit(rl_props=self.rl_props, engine_props=self.engine_props, is_path=False)
        else:
            raise NotImplementedError

        self.agent_obj.setup_env()

    def calculate_dynamic_penalty(self, core_index: float, req_id: float):
        """
        Calculate a dynamic penalty after every action.

        :param core_index: Core chosen.
        :param req_id: Current request ID.
        :return: The penalty that's calculated.
        :rtype: float
        """
        return self.engine_props['penalty'] * (1 + self.engine_props['gamma'] * core_index / req_id)

    def calculate_dynamic_reward(self, core_index: float, req_id: float):
        """
        Calculates a dynamic reward after every action.

        :param core_index: Core chosen.
        :param req_id: Current request ID.
        :return: The reward that's calculated.
        :rtype: float
        """
        core_decay = self.engine_props['reward'] / (1 + self.engine_props['decay_factor'] * core_index)
        request_weight = ((self.engine_props['num_requests'] - req_id) /
                          self.engine_props['num_requests']) ** self.engine_props['core_beta']

        return core_decay * request_weight

    def get_reward(self, was_allocated: bool):
        """
        Gets the core agent's reward based on the last action taken.

        :param was_allocated: If the last request was allocated.
        :return: The reward.
        :rtype: float
        """
        req_id = float(self.rl_help_obj.route_obj.sdn_props['req_id'])
        core_index = self.rl_props.core_index

        if was_allocated:
            if self.engine_props['dynamic_reward']:
                reward = self.calculate_dynamic_reward(core_index, req_id)
            else:
                reward = self.engine_props['reward']
            return reward

        if self.engine_props['dynamic_reward']:
            penalty = self.calculate_dynamic_penalty(core_index, req_id)
        else:
            penalty = self.engine_props['penalty']
        return penalty

    def update(self, was_allocated: bool, net_spec_dict: dict, iteration: int):
        """
        Makes updates to the core agent after each time step.

        :param was_allocated: If the request was allocated.
        :param net_spec_dict: The current network spectrum database.
        :param iteration: The current iteration
        """
        reward = self.get_reward(was_allocated=was_allocated)

        self.agent_obj.iteration = iteration
        if self.core_algorithm == 'q_learning':
            print('Core Index:', self.rl_props.core_index)
            self.agent_obj.update_cores_matrix(reward=reward, level_index=self.level_index,
                                               net_spec_dict=net_spec_dict, core_index=self.rl_props.core_index)
        elif self.core_algorithm == 'epsilon_greedy_bandit':
            self.agent_obj.update(reward=reward, arm=self.rl_props.core_index, iteration=iteration)
        elif self.core_algorithm == 'ucb_bandit':
            self.agent_obj.update(reward=reward, arm=self.rl_props.core_index, iteration=iteration)
        else:
            raise NotImplementedError

    def _ql_core(self):
        random_float = np.round(np.random.uniform(0, 1), decimals=1)
        cores_matrix = self.agent_obj.props.cores_matrix
        cores_matrix = cores_matrix[self.rl_props.source][self.rl_props.destination]
        self.rl_props.cores_list = cores_matrix[self.rl_props.chosen_path_index]
        self.cong_list = self.rl_help_obj.classify_cores(cores_list=self.rl_props.cores_list)

        if random_float < self.agent_obj.props.epsilon:
            self.rl_props.core_index = np.random.randint(0, self.engine_props['cores_per_link'])
            self.level_index = self.cong_list[self.rl_props.core_index][-1]
        else:
            self.rl_props.core_index, self.rl_props.chosen_core = self.agent_obj.get_max_curr_q(
                cong_list=self.cong_list,
                matrix_flag='cores_matrix')
            self.level_index = self.cong_list[self.rl_props.core_index][-1]

    def _bandit_core(self, path_index: int, source: str, dest: str):
        self.rl_props.core_index = self.agent_obj.select_core_arm(source=int(source), dest=int(dest),
                                                                  path_index=path_index)

    def get_core(self):
        """
        Assigns a core to the current request.
        """
        if self.core_algorithm == 'q_learning':
            self._ql_core()
        elif self.core_algorithm == 'epsilon_greedy_bandit':
            self._bandit_core(path_index=self.rl_props.chosen_path_index, source=self.rl_props.chosen_path_list[0][0],
                              dest=self.rl_props.chosen_path_list[0][-1])
        elif self.core_algorithm == 'ucb_bandit':
            self._bandit_core(path_index=self.rl_props.chosen_path_index, source=self.rl_props.chosen_path_list[0][0],
                              dest=self.rl_props.chosen_path_list[0][-1])
        else:
            raise NotImplementedError

    def load_model(self, model_path: str, erlang: float, num_cores: int):
        """
        Loads a previously trained core agent model.

        :param model_path: The path to the core agent model.
        :param erlang: The Erlang value the model was trained on.
        :param num_cores: The number of cores the model was trained on.
        """
        self.setup_env()
        if self.core_algorithm == 'q_learning':
            model_path = os.path.join('logs', model_path, f'e{erlang}_cores_c{num_cores}.npy')
            self.agent_obj.props.cores_matrix = np.load(model_path, allow_pickle=True)


# TODO: No longer supported/functional
class SpectrumAgent:
    """
    A class that handles everything related to spectrum assignment in reinforcement learning simulations.
    """

    def __init__(self, spectrum_algorithm: str, rl_props: object):
        self.spectrum_algorithm = spectrum_algorithm
        self.rl_props = rl_props

        self.no_penalty = None
        self.model = None

    def _ppo_obs_space(self):
        """
        Gets the observation space for the DRL agent.

        :return: The observation space.
        :rtype: spaces.Dict
        """
        resp_obs = spaces.Dict({
            'slots_needed': spaces.Discrete(15 + 1),
            'source': spaces.MultiBinary(self.rl_props.num_nodes),
            'destination': spaces.MultiBinary(self.rl_props.num_nodes),
            'super_channels': spaces.Box(-0.01, 100.0, shape=(3,), dtype=np.float32)
        })

        return resp_obs

    def get_obs_space(self):
        """
        Gets the observation space for each DRL model.

        :return: The DRL model's observation space.
        """
        if self.spectrum_algorithm == 'ppo':
            return self._ppo_obs_space()

        return None

    def _ppo_action_space(self):
        action_space = spaces.Discrete(self.rl_props.super_channel_space)
        return action_space

    def get_action_space(self):
        """
        Gets the action space for the DRL model.

        :return: The DRL model's action space.
        """
        if self.spectrum_algorithm == 'ppo':
            return self._ppo_action_space()

        return None

    def get_reward(self, was_allocated: bool):
        """
        Gets the reward for the spectrum agent.

        :param was_allocated: If the request was allocated or not.
        :return: The reward.
        :rtype: float
        """
        if self.no_penalty and not was_allocated:
            drl_reward = 0.0
        elif not was_allocated:
            drl_reward = -1.0
        else:
            drl_reward = 1.0

        return drl_reward


# TODO: Class needs to be used and updated properly
class HyperparamConfig:
    """
    Controls all hyperparameter starts, ends, and episodic and or time step modifications.
    """

    def __init__(self):
        self.total_steps = None
        self.time_step = None
        self.update_strategy = None

        self.alpha_start = None
        self.alpha_end = None
        self.curr_alpha = None

        self.epsilon_start = None
        self.epsilon_end = None
        self.curr_epsilon = None

        self.temperature = None
        self.state_visit_dict = None
        self.reward_dict = None
        self.q_table_dict = None
        self.decay_rate = None

    def _softmax(self, q_vals_list: list):
        """
        Compute the softmax probabilities for a given set of Q-values
        """
        exp_values = np.exp(np.array(q_vals_list) / self.temperature)
        probabilities = exp_values / np.sum(exp_values)
        return probabilities

    def _softmax_eps(self):
        """
        Softmax epsilon update rule.
        """
        # TODO: Update data structure
        q_vals_list = list(self.q_table_dict.values())
        softmax_probs = self._softmax(q_vals_list=q_vals_list)
        self.curr_epsilon = self.epsilon_start * np.sum(softmax_probs)

    def _softmax_alpha(self):
        """
        Softmax alpha update rule.
        """
        # TODO Update data structure
        q_vals_list = list(self.q_table_dict.values())
        softmax_probs = self._softmax(q_vals_list=q_vals_list)
        self.curr_alpha = self.alpha_start * np.sum(softmax_probs)

    def _reward_based_eps(self):
        """
        Reward-based epsilon update.
        """
        # TODO: Update data structure
        curr_reward = self.reward_dict["curr_reward"]
        last_reward = self.reward_dict["last_reward"]
        reward_diff = abs(curr_reward - last_reward)
        self.curr_epsilon = self.epsilon_start * (1 / (1 + reward_diff))

    def _reward_based_alpha(self):
        """
        Reward-based alpha update.
        """
        # TODO: Update data structure
        curr_reward = self.reward_dict["curr_reward"]
        last_reward = self.reward_dict["last_reward"]
        reward_diff = abs(curr_reward - last_reward)
        self.curr_alpha = self.alpha_start * (1 / (1 + reward_diff))

    def _state_based_eps(self):
        """
        State visitation epsilon update.
        """
        # TODO: Update data structure
        total_visits = sum(self.state_visit_dict.values())
        self.curr_epsilon = self.epsilon_start / (1 + total_visits)

    def _state_based_alpha(self):
        """
        State visitation alpha update.
        """
        # TODO: Update data structure
        total_visits = sum(self.state_visit_dict.values())
        self.curr_alpha = 1 / (1 + total_visits)

    def _exp_eps(self):
        """
        Exponential distribution epsilon update.
        """
        self.curr_epsilon = self.epsilon_start * (self.decay_rate ** self.time_step)

    def _exp_alpha(self):
        """
        Exponential distribution alpha update.
        """
        self.curr_alpha = self.alpha_start * (self.decay_rate ** self.time_step)

    def _linear_eps(self):
        """
        Linear decay epsilon update.
        """
        self.curr_epsilon = self.epsilon_end + (
                (self.epsilon_start - self.epsilon_end) * (self.total_steps - self.time_step) / self.total_steps
        )

    def _linear_alpha(self):
        """
        Linear decay alpha update.
        """
        self.curr_alpha = self.alpha_end + (
                (self.alpha_start - self.alpha_end) * (self.total_steps - self.time_step) / self.total_steps
        )

    def update_hyperparams(self):
        """
        Controls the class.
        """
        raise NotImplementedError

    def init_hyperparams(self):
        """
        Initializes this class.
        """
        raise NotImplementedError
