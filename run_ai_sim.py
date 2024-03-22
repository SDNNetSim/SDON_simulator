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
from helper_scripts.sim_helpers import get_start_time, find_core_frag_cong
from arg_scripts.ai_args import empty_dqn_props


class DQNSimEnv(gym.Env):  # pylint: disable=abstract-method
    """
    Simulates a deep q-learning environment with stable baselines3 integration.
    """
    metadata = dict()

    def __init__(self, render_mode: str = None, **kwargs):
        super().__init__()
        self.dqn_props = copy.deepcopy(empty_dqn_props)
        self.dqn_sim_dict = dict()
        self.engine_obj = None
        self.route_obj = None
        self.helper_obj = AIHelpers(ai_props=self.dqn_props, engine_obj=self.engine_obj)

        self.iteration = 0
        self.options = None
        self.k_paths = None
        self.cores_per_link = None
        self.spectral_slots = None
        self.num_nodes = -1
        self.min_arrival = np.inf
        self.max_arrival = -1 * np.inf
        self.min_depart = np.inf
        self.max_depart = -1 * np.inf
        self.bandwidth_list = list()

        # Used to get config variables into the observation space
        self.reset(options={'save_sim': False})
        self.max_slots_needed = 0
        self.max_length = 0
        self._find_maximums()

        self.observation_space = spaces.Dict({
            'source': spaces.Discrete(self.num_nodes, start=0),
            'destination': spaces.Discrete(self.num_nodes, start=0),
            'arrival': spaces.Box(low=-0.01, high=1.00, dtype=np.float64, shape=(1,)),
            'departure': spaces.Box(low=-0.01, high=1.00, dtype=np.float64, shape=(1,)),
            'bandwidth': spaces.MultiBinary(len(self.bandwidth_list)),
            # By two to represent a core's current fragmentation and congestion scores
            'cores_matrix': spaces.Box(low=0.01, high=1.01, shape=(self.k_paths, self.cores_per_link, 2)),
        })

        # 0 for no slicing, 1 for slicing
        self.slice_space = 2
        self.action_space = spaces.Discrete(self.k_paths * self.cores_per_link * self.slice_space)
        self.render_mode = render_mode

    def _find_maximums(self):
        for bandwidth, mod_obj in self.engine_obj.engine_props['mod_per_bw'].items():
            bandwidth_percent = self.engine_obj.engine_props['request_distribution'][bandwidth]
            if bandwidth_percent > 0:
                self.bandwidth_list.append(bandwidth)
            for modulation, data_obj in mod_obj.items():
                if data_obj['slots_needed'] > self.max_slots_needed and bandwidth_percent > 0:
                    self.max_slots_needed = data_obj['slots_needed']
                if data_obj['max_length'] > self.max_length and bandwidth_percent > 0:
                    self.max_length = data_obj['max_length']

    def _check_terminated(self):
        if self.dqn_props['arrival_count'] == (self.engine_obj.engine_props['num_requests']):
            terminated = True
            base_fp = os.path.join('data')
            self.engine_obj.end_iter(iteration=self.iteration, print_flag=False, ai_flag=True, base_fp=base_fp)
        else:
            terminated = False

        return terminated

    def _calculate_reward(self, was_allocated: bool):
        if was_allocated:
            reward = 1.0
        else:
            reward = -1.0

        return reward

    def _update_helper_obj(self, action: int):
        if self.engine_obj.engine_props['max_segments'] > 1:
            self.helper_obj.slice_request = action % self.slice_space
        else:
            self.helper_obj.slice_request = 0
        self.helper_obj.core_num = (action // self.slice_space) % self.cores_per_link
        self.helper_obj.path_index = action // (self.slice_space * self.cores_per_link)

        if self.helper_obj.path_index < 0 or self.helper_obj.path_index > (self.k_paths - 1):
            raise ValueError(f'Path index out of range: {self.helper_obj.path_index}')
        if self.helper_obj.core_num < 0 or self.helper_obj.core_num > (self.cores_per_link - 1):
            raise ValueError(f'Core index out of range: {self.helper_obj.core_num}')

        # TODO: Temporary
        self.helper_obj.ai_props = self.dqn_props
        self.helper_obj.engine_obj = self.engine_obj
        self.helper_obj.handle_releases()

    def _update_snapshots(self):
        arrival_count = self.dqn_props['arrival_count']

        snapshot_step = self.engine_obj.engine_props['snapshot_step']
        if self.engine_obj.engine_props['save_snapshots'] and (arrival_count + 1) % snapshot_step == 0:
            self.engine_obj.stats_obj.update_snapshot(net_spec_dict=self.engine_obj.net_spec_dict,
                                                      req_num=arrival_count + 1)

    def step(self, action: int):
        self._update_helper_obj(action=action)
        self.helper_obj.allocate(route_obj=self.route_obj)
        reqs_status_dict = self.engine_obj.reqs_status_dict

        req_id = self.dqn_props['arrival_list'][self.dqn_props['arrival_count']]['req_id']
        if req_id in reqs_status_dict:
            was_allocated = True
        else:
            was_allocated = False
        self._update_snapshots()

        reward = self._calculate_reward(was_allocated=was_allocated)
        self.dqn_props['arrival_count'] += 1
        terminated = self._check_terminated()
        new_obs = self._get_obs()
        truncated = False
        info = self._get_info()

        return new_obs, reward, terminated, truncated, info

    def _get_spectrum(self, paths_matrix: list):
        # To add core and fragmentation scores, make a k_path by cores by two matrix (two metrics)
        spectrum_matrix = np.zeros((self.k_paths, self.cores_per_link, 2))
        for path_index, paths_list in enumerate(paths_matrix):
            for link_tuple in zip(paths_list, paths_list[1:]):
                rev_link_tuple = link_tuple[1], link_tuple[0]
                link_dict = self.engine_obj.net_spec_dict[link_tuple]
                rev_link_dict = self.engine_obj.net_spec_dict[rev_link_tuple]

                if link_dict != rev_link_dict:
                    raise ValueError('Link is not bi-directionally equal.')

                for core_index, core_arr in enumerate(link_dict['cores_matrix']):
                    spectrum_matrix[path_index][core_index] = find_core_frag_cong(
                        net_spec_db=self.engine_obj.net_spec_dict, path=paths_list, core=core_index)

        return spectrum_matrix

    @staticmethod
    def _get_info():
        return dict()

    # TODO: Move to sim functions
    @staticmethod
    def _min_max_scale(value: float, min_value: float, max_value: float):
        return (value - min_value) / (max_value - min_value)

    def _get_obs(self):
        # Used when we reach a reset after a simulation has finished (reset automatically called by gymnasium, use
        # placeholder variable)
        if self.dqn_props['arrival_count'] == self.engine_obj.engine_props['num_requests']:
            curr_req = self.dqn_props['arrival_list'][self.dqn_props['arrival_count'] - 1]
        else:
            curr_req = self.dqn_props['arrival_list'][self.dqn_props['arrival_count']]

        self.dqn_props['mock_sdn_dict'] = self.helper_obj.update_mock_sdn(curr_req=curr_req)
        self.route_obj.sdn_props = self.dqn_props['mock_sdn_dict']
        self.route_obj.get_route()

        paths_matrix = self.route_obj.route_props['paths_list']
        spectrum_obs = self._get_spectrum(paths_matrix=paths_matrix)
        arrival_scaled = self._min_max_scale(value=curr_req['arrive'], min_value=self.min_arrival,
                                             max_value=self.max_arrival)
        depart_scaled = self._min_max_scale(value=curr_req['depart'], min_value=self.min_depart,
                                            max_value=self.max_depart)

        encode_bw_list = np.zeros((3,))
        if len(self.bandwidth_list) != 0:
            bandwidth_index = self.bandwidth_list.index(curr_req['bandwidth'])
            encode_bw_list[bandwidth_index] = 1

        obs_dict = {
            'source': int(curr_req['source']),
            'destination': int(curr_req['destination']),
            'bandwidth': encode_bw_list,
            'arrival': np.array([arrival_scaled]),
            'departure': np.array([depart_scaled]),
            'cores_matrix': spectrum_obs
        }
        return obs_dict

    def _reset_reqs_dict(self, seed: int):
        self.engine_obj.generate_requests(seed=seed)
        self.min_arrival = np.inf
        self.max_arrival = -1 * np.inf
        self.min_depart = np.inf
        self.max_depart = -1 * np.inf

        for req_time in self.engine_obj.reqs_dict:
            if self.engine_obj.reqs_dict[req_time]['request_type'] == 'arrival':
                if req_time > self.max_arrival:
                    self.max_arrival = req_time
                if req_time < self.min_arrival:
                    self.min_arrival = req_time

                self.dqn_props['arrival_list'].append(self.engine_obj.reqs_dict[req_time])
            else:
                if req_time > self.max_depart:
                    self.max_depart = req_time
                if req_time < self.min_depart:
                    self.min_depart = req_time

                self.dqn_props['depart_list'].append(self.engine_obj.reqs_dict[req_time])

    def _create_input(self):
        base_fp = os.path.join('data')
        self.dqn_sim_dict['s1']['thread_num'] = 's1'
        get_start_time(sim_dict=self.dqn_sim_dict)
        file_name = "sim_input_s1.json"

        self.engine_obj = Engine(engine_props=self.dqn_sim_dict['s1'])
        self.route_obj = Routing(engine_props=self.engine_obj.engine_props, sdn_props=self.dqn_props['mock_sdn_dict'])
        self.dqn_sim_dict['s1'] = create_input(base_fp=base_fp, engine_props=self.dqn_sim_dict['s1'])

        if self.options['save_sim']:
            save_input(base_fp=base_fp, properties=self.dqn_sim_dict['s1'], file_name=file_name,
                       data_dict=self.dqn_sim_dict['s1'])

    def setup(self):
        """
        Sets up this class.
        """
        args_obj = parse_args()
        config_path = os.path.join('ini', 'run_ini', 'config.ini')
        self.dqn_sim_dict = read_config(args_obj=args_obj, config_path=config_path)

        self.k_paths = self.dqn_sim_dict['s1']['k_paths']
        self.cores_per_link = self.dqn_sim_dict['s1']['cores_per_link']
        self.spectral_slots = self.dqn_sim_dict['s1']['spectral_slots']

        self._create_input()
        start_arr_rate = float(self.dqn_sim_dict['s1']['arrival_rate']['start'])
        self.engine_obj.engine_props['erlang'] = start_arr_rate / self.dqn_sim_dict['s1']['holding_time']
        self.engine_obj.engine_props['arrival_rate'] = start_arr_rate * self.dqn_sim_dict['s1']['cores_per_link']

    def reset(self, seed: int = None, options: dict = None):  # pylint: disable=arguments-differ
        # Default options
        if options is None:
            self.options = {'save_sim': True}
        else:
            self.options = options

        super().reset(seed=seed)
        self.dqn_props = copy.deepcopy(empty_dqn_props)
        self.setup()
        self.dqn_props['arrival_count'] = 0
        self.engine_obj.init_iter(iteration=self.iteration)
        self.engine_obj.create_topology()
        self.num_nodes = len(self.engine_obj.topology.nodes)

        if seed is None:
            seed = self.iteration
        self._reset_reqs_dict(seed=seed)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info


if __name__ == '__main__':
    env = DQNSimEnv()

    # model = DQN("MultiInputPolicy", env, verbose=1)
    # model.learn(total_timesteps=10000, log_interval=1)

    # model = DQN.load('./logs/dqn/DQNSimEnv_5/best_model.zip', env=env)
    # obs, info = env.reset()
    # episode_reward = 0
    # max_episodes = 5
    # num_episodes = 0
    # while True:
    #     curr_action, _states = model.predict(obs)
    #
    #     obs, curr_reward, is_terminated, is_truncated, curr_info = env.step(curr_action)
    #     episode_reward += curr_reward
    #     if num_episodes >= max_episodes:
    #         break
    #     if is_terminated or is_truncated:
    #         obs, info = env.reset()
    #         num_episodes += 1
    #
    # print(episode_reward / max_episodes)
    # obs, info = env.reset()
