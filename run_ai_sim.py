import os
import copy

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import DQN

from config_scripts.parse_args import parse_args
from config_scripts.setup_config import read_config
from sim_scripts.engine import Engine
from sim_scripts.routing import Routing
from helper_scripts.setup_helpers import create_input, save_input
from helper_scripts.ai_helpers import AIHelpers
from helper_scripts.sim_helpers import combine_and_one_hot, get_start_time
from arg_scripts.ai_args import empty_dqn_props


# TODO: Not supported:
#   - Threading
#   - Segment slicing
#   - Path selection

class DQNSimEnv(gym.Env):  # pylint: disable=abstract-method
    """
    Simulates a deep q-learning environment with stable baselines3 integration.
    """
    metadata = dict()

    def __init__(self, render_mode: str = None, **kwargs):
        super().__init__()
        self.dqn_props = copy.deepcopy(empty_dqn_props)
        self.dqn_sim_dict = None
        self.engine_obj = None
        self.route_obj = None
        self.helper_obj = AIHelpers(ai_props=self.dqn_props)

        self.k_paths = kwargs['arguments'][0]
        self.cores_per_link = kwargs['arguments'][1]
        self.spectral_slots = kwargs['arguments'][2]
        self.num_requests = kwargs['arguments'][3]

        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(self.k_paths, self.cores_per_link, self.spectral_slots),
                                            dtype=np.float64)
        self.action_space = spaces.Discrete(self.k_paths * self.cores_per_link * self.spectral_slots)
        self.render_mode = render_mode
        self.iteration = 0

    def _check_terminated(self):
        if self.dqn_props['arrival_count'] == (self.dqn_props['engine_props']['num_requests']):
            terminated = True
            base_fp = os.path.join('data')
            self.engine_obj.end_iter(iteration=self.iteration, print_flag=False, ai_flag=True, base_fp=base_fp)
            self.iteration += 1
        else:
            terminated = False

        return terminated

    @staticmethod
    def _calculate_reward(was_allocated: bool):
        if was_allocated:
            reward = 1.0
        else:
            reward = -1.0

        return reward

    def _update_helper_obj(self, action: int):
        self.helper_obj.path_index = action // (self.cores_per_link * self.spectral_slots)
        remaining = action % (self.cores_per_link * self.spectral_slots)
        self.helper_obj.core_num = remaining // self.spectral_slots
        self.helper_obj.start_slot = remaining % self.spectral_slots

        self.helper_obj.check_release()

        self.helper_obj.net_spec_dict = self.engine_obj.net_spec_dict
        self.helper_obj.reqs_status_dict = self.engine_obj.reqs_status_dict

    def step(self, action: int):
        self._update_helper_obj(action=action)
        was_allocated = self.helper_obj.allocate(route_obj=self.route_obj)
        curr_time = self.dqn_props['arrival_list'][self.dqn_props['arrival_count']]['arrive']
        self.engine_obj.update_arrival_params(curr_time=curr_time, ai_flag=True,
                                              mock_sdn=self.dqn_props['mock_sdn_dict'])

        reward = self._calculate_reward(was_allocated=was_allocated)
        self.dqn_props['arrival_count'] += 1
        terminated = self._check_terminated()
        new_obs = self._get_obs()
        truncated = False
        info = self._get_info()

        return new_obs, reward, terminated, truncated, info

    def _get_spectrum(self, paths_matrix: list):
        spectrum_matrix = np.zeros((self.k_paths, self.cores_per_link, self.spectral_slots))
        for path_index, paths_list in enumerate(paths_matrix):
            for link_tuple in zip(paths_list, paths_list[1:]):
                rev_link_tuple = link_tuple[1], link_tuple[0]
                link_dict = self.engine_obj.net_spec_dict[link_tuple]
                rev_link_dict = self.engine_obj.net_spec_dict[rev_link_tuple]

                if link_dict != rev_link_dict:
                    raise ValueError('Link is not bi-directionally equal.')

                for core_index, core_arr in enumerate(link_dict['cores_matrix']):
                    spectrum_matrix[path_index][core_index] = combine_and_one_hot(
                        arr1=spectrum_matrix[path_index][core_index],
                        arr2=core_arr
                    )

        return spectrum_matrix

    @staticmethod
    def _get_info():
        return dict()

    def _get_obs(self):
        if self.dqn_props['arrival_count'] == self.dqn_props['engine_props']['num_requests']:
            curr_req = self.dqn_props['arrival_list'][self.dqn_props['arrival_count'] - 1]
        else:
            curr_req = self.dqn_props['arrival_list'][self.dqn_props['arrival_count']]

        self.helper_obj.topology = self.dqn_props['engine_props']['topology']
        self.dqn_props['mock_sdn_dict'] = self.helper_obj.update_mock_sdn(curr_req=curr_req)

        self.route_obj.sdn_props = self.dqn_props['mock_sdn_dict']
        self.route_obj.get_route()

        paths_matrix = self.route_obj.route_props['paths_list']
        spectrum_obs = self._get_spectrum(paths_matrix=paths_matrix)
        return spectrum_obs

    def _create_input(self):
        base_fp = os.path.join('data')
        self.dqn_sim_dict['s1']['thread_num'] = 's1'
        get_start_time(sim_dict=self.dqn_sim_dict)
        file_name = "sim_input_s1.json"

        self.engine_obj = Engine(engine_props=self.dqn_sim_dict['s1'])
        self.route_obj = Routing(engine_props=self.engine_obj.engine_props, sdn_props=self.dqn_props['mock_sdn_dict'])
        self.dqn_sim_dict['s1'] = create_input(base_fp=base_fp, engine_props=self.dqn_sim_dict['s1'])

        save_input(base_fp=base_fp, properties=self.dqn_sim_dict['s1'], file_name=file_name,
                   data_dict=self.dqn_sim_dict['s1'])

    def setup(self):
        """
        Sets up this class.
        """
        args_obj = parse_args()
        config_path = os.path.join('ini', 'run_ini', 'config.ini')
        self.dqn_sim_dict = read_config(args_obj=args_obj, config_path=config_path)

        self.dqn_sim_dict['s1']['cores_per_link'] = self.cores_per_link
        self.dqn_sim_dict['s1']['spectral_slots'] = self.spectral_slots
        self.dqn_sim_dict['s1']['num_requests'] = self.num_requests
        self._create_input()
        start_arr_rate = float(self.dqn_sim_dict['s1']['arrival_rate']['start'])
        self.engine_obj.engine_props['erlang'] = start_arr_rate / self.dqn_sim_dict['s1']['holding_time']
        self.engine_obj.engine_props['arrival_rate'] = start_arr_rate * self.dqn_sim_dict['s1']['cores_per_link']

    def _reset_reqs_dict(self, seed: int):
        self.engine_obj.generate_requests(seed=seed)
        for req_time in self.engine_obj.reqs_dict:
            if self.engine_obj.reqs_dict[req_time]['request_type'] == 'arrival':
                self.dqn_props['arrival_list'].append(self.engine_obj.reqs_dict[req_time])
            else:
                self.dqn_props['depart_list'].append(self.engine_obj.reqs_dict[req_time])

        self.dqn_props['reqs_dict'] = self.engine_obj.reqs_dict

    def reset(self, seed: int = None, options: dict = None):  # pylint: disable=arguments-differ
        super().reset(seed=seed)
        self.dqn_props = copy.deepcopy(empty_dqn_props)
        self.helper_obj.ai_props = self.dqn_props
        self.setup()
        self.dqn_props['arrival_count'] = 0
        self.dqn_props['engine_props'] = self.engine_obj.engine_props
        self.engine_obj.init_iter(iteration=self.iteration)
        self.engine_obj.create_topology()

        if seed is None:
            seed = self.iteration
        self._reset_reqs_dict(seed=seed)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info


if __name__ == '__main__':
    env = DQNSimEnv()

    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=40, log_interval=5)

    # obs, info = env.reset()
    # while True:
    #     curr_action, _states = model.predict(obs, deterministic=True)
    #
    #     obs, curr_reward, is_terminated, is_truncated, curr_info = env.step(curr_action)
    #     if is_terminated or is_truncated:
    #         break
    # obs, info = env.reset()
