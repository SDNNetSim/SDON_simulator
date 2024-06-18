import os
import json

import numpy as np
from gymnasium import spaces

from .sim_helpers import find_path_cong, find_core_cong
from .ql_helpers import QLearningHelpers
from .bandit_helpers import EpsilonGreedyBandit, ContextualEpsilonGreedyBandit, ContextGenerator
from .bandit_helpers import ThompsonSamplingBandit, UCBBandit


class PathAgent:
    """
    A class that handles everything related to path assignment in reinforcement learning simulations.
    """

    def __init__(self, path_algorithm: str, rl_props: dict, rl_help_obj: object):
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
        elif self.path_algorithm == 'epsilon_greedy_bandit':
            self.agent_obj = EpsilonGreedyBandit(rl_props=self.rl_props, engine_props=self.engine_props, is_path=True,
                                                 is_core=False)
        elif self.path_algorithm == 'context_epsilon_greedy_bandit':
            self.agent_obj = ContextualEpsilonGreedyBandit(rl_props=self.rl_props, engine_props=self.engine_props)
            self.context_obj = ContextGenerator(rl_props=self.rl_props, engine_props=self.engine_props)
        elif self.path_algorithm == 'ucb_bandit':
            self.agent_obj = UCBBandit(rl_props=self.rl_props, engine_props=self.engine_props)
        elif self.path_algorithm == 'thompson_sampling_bandit':
            self.agent_obj = ThompsonSamplingBandit(rl_props=self.rl_props, engine_props=self.engine_props)
        else:
            raise NotImplementedError

        self.agent_obj.setup_env()

    def get_reward(self, was_allocated: bool, path_length: int):
        """
        Get the current reward for the last agent's action.

        :param was_allocated: If the request was allocated or not.
        :param path_length: The path length.
        :return: The reward.
        :rtype: float
        """
        # TODO: Incorporate or delete
        beta = 0.0
        if was_allocated:
            return self.engine_props['reward']

        return (self.engine_props['penalty'] * (1 - beta)) - (path_length * beta)

    def update(self, was_allocated: bool, net_spec_dict: dict, iteration: int, path_length):
        """
        Makes updates to the agent for each time step.

        :param was_allocated: If the request was allocated.
        :param net_spec_dict: The current network spectrum database.
        :param iteration: The current iteration.
        """
        reward = self.get_reward(was_allocated=was_allocated, path_length=path_length)

        self.agent_obj.iteration = iteration
        if self.path_algorithm == 'q_learning':
            self.agent_obj.update_routes_matrix(reward=reward, level_index=self.level_index,
                                                net_spec_dict=net_spec_dict)
        elif self.path_algorithm == 'epsilon_greedy_bandit':
            self.agent_obj.update(reward=reward, arm=self.rl_props['chosen_path_index'], iteration=iteration)
        elif self.path_algorithm == 'ucb_bandit':
            self.agent_obj.update(reward=reward, arm=self.rl_props['chosen_path_index'])
        elif self.path_algorithm == 'thompson_sampling_bandit':
            self.agent_obj.update(reward=reward, arm=self.rl_props['chosen_path_index'])
        elif self.path_algorithm == 'context_epsilon_greedy_bandit':
            self.agent_obj.update(reward=reward, arm=self.rl_props['chosen_path_index'],
                                  context=self.context_obj.curr_context)
        else:
            raise NotImplementedError

    def __ql_route(self, random_float: float):
        if random_float < self.agent_obj.props['epsilon']:
            self.rl_props['path_index'] = np.random.choice(self.rl_props['k_paths'])
            # The level will always be the last index
            self.level_index = self.cong_list[self.rl_props['path_index']][-1]

            if self.rl_props['path_index'] == 1 and self.rl_props['k_paths'] == 1:
                self.rl_props['path_index'] = 0
            self.rl_props['chosen_path'] = self.rl_props['paths_list'][self.rl_props['path_index']]
        else:
            self.rl_props['path_index'], self.rl_props['chosen_path'] = self.agent_obj.get_max_curr_q(
                cong_list=self.cong_list, matrix_flag='routes_matrix')
            self.level_index = self.cong_list[self.rl_props['path_index']][-1]

    def _ql_route(self):
        random_float = float(np.round(np.random.uniform(0, 1), decimals=1))
        routes_matrix = self.agent_obj.props['routes_matrix']
        self.rl_props['paths_list'] = routes_matrix[self.rl_props['source']][self.rl_props['destination']]['path']

        self.cong_list = self.rl_help_obj.classify_paths(paths_list=self.rl_props['paths_list'])
        if self.rl_props['paths_list'].ndim != 1:
            self.rl_props['paths_list'] = self.rl_props['paths_list'][:, 0]

        self.__ql_route(random_float=random_float)

        if len(self.rl_props['chosen_path']) == 0:
            raise ValueError('The chosen path can not be None')

    # TODO: Ideally q-learning should be like this (agent_obj.something)
    #   - Need access to the actual path
    def _bandit_route(self, route_obj: object):
        paths_list = route_obj.route_props['paths_list']
        source = paths_list[0][0]
        dest = paths_list[0][-1]
        self.rl_props['chosen_path_index'] = self.agent_obj.select_path_arm(source=int(source), dest=int(dest))
        self.rl_props['chosen_path'] = route_obj.route_props['paths_list'][self.rl_props['chosen_path_index']]

    def _context_bandit_route(self, route_obj: object):
        cong_list = list()
        source = None
        dest = None
        # TODO: Make sure net_spec_db is correct and updated
        for path_list in route_obj.route_props['paths_list']:
            net_spec_dict = self.rl_help_obj.engine_obj.net_spec_dict
            curr_cong = find_path_cong(path_list=path_list, net_spec_dict=net_spec_dict)
            cong_list.append(curr_cong)

            if source is None:
                source = int(path_list[0])
                dest = int(path_list[-1])

        # TODO: Make sure converting to int doesn't cause any discrepancies
        self.context_obj.generate_context(source=source, dest=dest, congestion_levels=cong_list)
        self.rl_props['chosen_path_index'] = self.agent_obj.select_arm(context=self.context_obj.curr_context,
                                                                       source=source, dest=dest)
        self.rl_props['chosen_path'] = route_obj.route_props['paths_list'][self.rl_props['chosen_path_index']]

    def get_route(self, **kwargs):
        """
        Assign a route for the current request.
        """
        if self.path_algorithm == 'q_learning':
            self._ql_route()
        elif self.path_algorithm in ('epsilon_greedy_bandit', 'thompson_sampling_bandit', 'ucb_bandit'):
            self._bandit_route(route_obj=kwargs['route_obj'])
        elif self.path_algorithm == 'context_epsilon_greedy_bandit':
            self._context_bandit_route(route_obj=kwargs['route_obj'])
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
            self.agent_obj.props['routes_matrix'] = np.load(model_path, allow_pickle=True)
        else:
            pass


class CoreAgent:
    """
    A class that handles everything related to core assignment in reinforcement learning simulations.
    """

    def __init__(self, core_algorithm: str, rl_props: dict, rl_help_obj: object):
        self.core_algorithm = core_algorithm
        self.rl_props = rl_props
        self.engine_props = None
        self.agent_obj = None
        self.rl_help_obj = rl_help_obj

        self.level_index = None
        self.cong_list = list()
        self.no_penalty = False

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
            self.agent_obj = EpsilonGreedyBandit(rl_props=self.rl_props, engine_props=self.engine_props, is_path=False,
                                                 is_core=True)
        else:
            raise NotImplementedError

        self.agent_obj.setup_env()

    def calculate_dynamic_penalty(self, core_index, req_id):
        return self.engine_props['penalty'] * (1 + self.engine_props['gamma'] * core_index / req_id)

    def calculate_dynamic_reward(self, core_index, req_id):
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
        req_id = float(self.rl_help_obj.sdn_props['req_id'])
        core_index = self.rl_props['core_index']

        if was_allocated:
            reward = self.calculate_dynamic_reward(core_index, req_id)
            return reward
        else:
            penalty = self.calculate_dynamic_penalty(core_index, req_id)
            return penalty

    def update(self, was_allocated: bool, net_spec_dict: dict, iteration: int):
        """
        Makes updates to the core agent after each time step.

        :param was_allocated: If the request was allocated.
        :param net_spec_dict: The current network spectrum database.
        :param iteration: The current iteration
        """
        reward = self.get_reward(was_allocated=was_allocated, net_spec_dict=net_spec_dict)

        self.agent_obj.iteration = iteration
        if self.core_algorithm == 'q_learning':
            self.agent_obj.update_cores_matrix(reward=reward, level_index=self.level_index,
                                               net_spec_dict=net_spec_dict, core_index=self.rl_props['core_index'])
        elif self.core_algorithm == 'epsilon_greedy_bandit':
            self.agent_obj.update(reward=reward, arm=self.rl_props['core_index'], iteration=iteration)
        else:
            raise NotImplementedError

    def _ql_core(self):
        random_float = np.round(np.random.uniform(0, 1), decimals=1)
        cores_matrix = self.agent_obj.props['cores_matrix']
        cores_matrix = cores_matrix[self.rl_props['source']][self.rl_props['destination']]
        self.rl_props['cores_list'] = cores_matrix[self.rl_props['path_index']]
        self.cong_list = self.rl_help_obj.classify_cores(cores_list=self.rl_props['cores_list'])

        if random_float < self.agent_obj.props['epsilon']:
            self.rl_props['core_index'] = np.random.randint(0, self.engine_props['cores_per_link'])
            self.level_index = self.cong_list[self.rl_props['core_index']][-1]
        else:
            self.rl_props['core_index'], self.rl_props['chosen_core'] = self.agent_obj.get_max_curr_q(
                cong_list=self.cong_list,
                matrix_flag='cores_matrix')
            self.level_index = self.cong_list[self.rl_props['core_index']][-1]

    def _bandit_core(self, path_index: int, source: str, dest: str):
        self.rl_props['core_index'] = self.agent_obj.select_core_arm(source=int(source), dest=int(dest),
                                                                     path_index=path_index)

    def get_core(self):
        """
        Assigns a core to the current request.
        """
        if self.core_algorithm == 'q_learning':
            self._ql_core()
        elif self.core_algorithm == 'epsilon_greedy_bandit':
            self._bandit_core(path_index=self.rl_props['chosen_path_index'], source=self.rl_props['chosen_path'][0][0],
                              dest=self.rl_props['chosen_path'][0][-1])
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
            self.agent_obj.props['cores_matrix'] = np.load(model_path, allow_pickle=True)


class SpectrumAgent:
    """
    A class that handles everything related to spectrum assignment in reinforcement learning simulations.
    """

    def __init__(self, spectrum_algorithm: str, rl_props: dict):
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
        # TODO: Change, hard coded
        resp_obs = spaces.Dict({
            'slots_needed': spaces.Discrete(15 + 1),
            'source': spaces.MultiBinary(self.rl_props['num_nodes']),
            'destination': spaces.MultiBinary(self.rl_props['num_nodes']),
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
        action_space = spaces.Discrete(self.rl_props['super_channel_space'])
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
