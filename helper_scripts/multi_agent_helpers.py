import numpy as np

from gymnasium import spaces
from .ql_helpers import QLearningHelpers


class PathAgent:
    def __init__(self, path_algorithm: str, rl_props: dict, rl_help_obj: object):
        self.path_algorithm = path_algorithm
        self.engine_props = None
        self.rl_props = rl_props
        self.rl_help_obj = rl_help_obj

        self.agent_obj = None

    def end_iter(self):
        if self.path_algorithm == 'q_learning':
            self.agent_obj.decay_epsilon()

    def setup_env(self):
        if self.path_algorithm == 'q_learning':
            self.agent_obj = QLearningHelpers(rl_props=self.rl_props, engine_props=self.engine_props)
        else:
            raise NotImplementedError

        self.agent_obj.setup_env()

    def get_reward(self, was_allocated: bool):
        if was_allocated:
            return 1.0
        else:
            return -1.0

    def update(self, was_allocated: bool, net_spec_dict: dict, iteration: int):
        reward = self.get_reward(was_allocated=was_allocated)

        if self.path_algorithm == 'q_learning':
            self.agent_obj.iteration = iteration
            self.agent_obj.update_routes_matrix(reward=reward, level_index=self.level_index,
                                                net_spec_dict=net_spec_dict)

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

    def get_route(self):
        if self.path_algorithm == 'q_learning':
            self._ql_route()
        else:
            raise NotImplementedError

    def load_model(self, model_path: str):
        # TODO: Hard coded, also need Erlang here, also params
        route_matrix_path = f'logs/ql/{model_path}/e250.0_routes_c7.npy'
        self.agent_obj.props['routes_matrix'] = np.load(route_matrix_path, allow_pickle=True)


class CoreAgent:
    def __init__(self, core_algorithm: str, rl_props: dict, rl_help_obj: object):
        self.core_algorithm = core_algorithm
        self.rl_props = rl_props
        self.engine_props = None
        self.agent_obj = None
        self.rl_help_obj = rl_help_obj

        self.level_index = None
        self.cong_list = list()

    def end_iter(self):
        # TODO: Only save core/path algorithm
        if self.core_algorithm == 'q_learning':
            self.agent_obj.decay_epsilon()

    def setup_env(self):
        if self.core_algorithm == 'q_learning':
            self.agent_obj = QLearningHelpers(rl_props=self.rl_props, engine_props=self.engine_props)
        else:
            raise NotImplementedError

        self.agent_obj.setup_env()

    @staticmethod
    def get_reward(was_allocated: bool):
        if was_allocated:
            return 1.0
        else:
            return -1.0

    def update(self, was_allocated: bool, net_spec_dict: dict, iteration: int):
        reward = self.get_reward(was_allocated=was_allocated)

        if self.core_algorithm == 'q_learning':
            self.agent_obj.iteration = iteration
            self.agent_obj.update_cores_matrix(reward=reward, level_index=self.level_index,
                                               net_spec_dict=net_spec_dict, core_index=self.rl_props['core_index'])

    def get_obs(self):
        raise NotImplementedError

    def get_action(self):
        raise NotImplementedError

    def _ql_core(self):
        random_float = np.round(np.random.uniform(0, 1), decimals=1)
        cores_matrix = self.agent_obj.props['cores_matrix']
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

    def get_core(self):
        if self.core_algorithm == 'q_learning':
            self._ql_core()

    def load_model(self, model_path: str):
        # TODO: Hard coded, also need Erlang here, also params
        core_matrix_path = f'logs/ql/{model_path}/e250.0_cores_c7.npy'
        self.agent_obj.props['cores_matrix'] = np.load(core_matrix_path, allow_pickle=True)


# The spectrum was almost to maximum capacity, there will be blocking but it's not the agent's fault
# Put the start index to zero (which will block regardless of what it is), but don't penalize the agent
# if self.no_penalty:
#     start_index = 0
# else:
#     start_index = self.super_channel_indexes[self.super_channel][0]
class SpectrumAgent:
    def __init__(self, spectrum_algorithm: str, rl_props: dict):
        self.spectrum_algorithm = spectrum_algorithm
        self.rl_props = rl_props

    # TODO: Change, hard-coded
    def _ppo_obs_space(self):
        """
        Gets the observation space for the DRL agent.

        :return: The observation space.
        :rtype: spaces.Dict
        """
        resp_obs = spaces.Dict({
            'slots_needed': spaces.Discrete(15 + 1),
            'source': spaces.MultiBinary(self.rl_props['num_nodes']),
            'destination': spaces.MultiBinary(self.rl_props['num_nodes']),
            'super_channels': spaces.Box(-0.01, 100.0, shape=(3,), dtype=np.float32)
        })

        return resp_obs

    def get_obs_space(self):
        if self.spectrum_algorithm == 'ppo':
            return self._ppo_obs_space()

    # TODO: Change hard-coded
    @staticmethod
    def _ppo_action_space(super_channel_space: int = 3):
        """
        Gets the action space for the DRL agent.

        :param super_channel_space: The number of 'J' super-channels that can be selected.
        :return: The action space.
        :rtype: spaces.Discrete
        """
        action_space = spaces.Discrete(super_channel_space)
        return action_space

    def get_action_space(self):
        if self.spectrum_algorithm == 'ppo':
            return self._ppo_obs_space()

    @staticmethod
    def _calc_deep_reward(was_allocated: bool):
        if was_allocated:
            reward = 1.0
        else:
            reward = -1.0

        return reward

    def calculate_drl_reward(self, was_allocated: bool):
        """
        Gets the reward for the deep reinforcement learning agent.

        :param was_allocated: Determines if the request was successfully allocated or not.
        :return: The reward.
        :rtype: float
        """
        if self.no_penalty:
            drl_reward = 0.0
        else:
            drl_reward = self._calc_deep_reward(was_allocated=was_allocated)

        return drl_reward

    def get_reward(self):
        raise NotImplementedError

    def _ppo_spectrum(self):
        """
        Returns the spectrum as a binary array along a path.
        A one indicates that channel is taken along one or multiple of the links, a zero indicates that the channel
        is free along every link in the path.

        :return: The binary array of current path occupation.
        :rtype: list
        """
        resp_spec_arr = np.zeros(self.engine_obj.engine_props['spectral_slots'])
        path_list = self.ai_props['paths_list'][self.ai_props['path_index']]
        core_index = self.ai_props['core_index']
        net_spec_dict = self.engine_obj.net_spec_dict
        for source, dest in zip(path_list, path_list[1:]):
            core_arr = net_spec_dict[(source, dest)]['cores_matrix'][core_index]
            resp_spec_arr = combine_and_one_hot(resp_spec_arr, core_arr)

        return resp_spec_arr

    def get_spectrum(self):
        raise NotImplementedError
