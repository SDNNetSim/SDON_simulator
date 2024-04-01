import os
import copy

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from config_scripts.parse_args import parse_args
from config_scripts.setup_config import read_config
from sim_scripts.engine import Engine
from sim_scripts.routing import Routing
from helper_scripts.setup_helpers import create_input, save_input
from helper_scripts.ai_helpers import AIHelpers
from helper_scripts.sim_helpers import get_start_time
from arg_scripts.ai_args import empty_dqn_props, empty_q_props


class SimEnv(gym.Env):  # pylint: disable=abstract-method
    """
    Simulates a deep q-learning environment with stable baselines3 integration.
    """
    metadata = dict()

    def __init__(self, render_mode: str = None, **kwargs):
        super().__init__()
        # TODO: Kwargs needs to be updated (env args by command line)
        if kwargs['algorithm'] in ('dqn', 'ppo'):
            self.ai_props = copy.deepcopy(empty_dqn_props)
        elif kwargs['algorithm'] == 'q_learning':
            self.ai_props = copy.deepcopy(empty_q_props)
        else:
            raise NotImplementedError
        self.algorithm = kwargs['algorithm']

        self.sim_dict = dict()
        self.iteration = 0
        self.options = None
        self.engine_obj = None
        self.route_obj = None
        self.helper_obj = AIHelpers(ai_props=self.ai_props, engine_obj=self.engine_obj, route_obj=self.route_obj)
        # Used to get config variables into the observation space
        self.reset(options={'save_sim': False})
        if kwargs['algorithm'] in ('dqn', 'ppo'):
            self.helper_obj.find_maximums()

        self.observation_space = self.helper_obj.get_obs_space(algorithm=kwargs['algorithm'])
        self.action_space = self.helper_obj.get_action_space(algorithm=kwargs['algorithm'])
        self.render_mode = render_mode

    def _check_terminated(self):
        if self.ai_props['arrival_count'] == (self.engine_obj.engine_props['num_requests']):
            terminated = True
            base_fp = os.path.join('data')
            self.engine_obj.end_iter(iteration=self.iteration, print_flag=False, ai_flag=True, base_fp=base_fp)
        else:
            terminated = False

        return terminated

    def _update_helper_obj(self, action: list):
        if self.algorithm in ('dqn', 'ppo'):
            self.helper_obj.path_index = action[0]
            self.helper_obj.core_num = action[1]
            self.helper_obj.slice_request = action[2]
        else:
            raise NotImplementedError

        if self.helper_obj.path_index < 0 or self.helper_obj.path_index > (self.ai_props['k_paths'] - 1):
            raise ValueError(f'Path index out of range: {self.helper_obj.path_index}')
        if self.helper_obj.core_num < 0 or self.helper_obj.core_num > (self.ai_props['cores_per_link'] - 1):
            raise ValueError(f'Core index out of range: {self.helper_obj.core_num}')

        self.helper_obj.ai_props = self.ai_props
        self.helper_obj.engine_obj = self.engine_obj
        self.helper_obj.handle_releases()

    def _update_snapshots(self):
        arrival_count = self.ai_props['arrival_count']

        snapshot_step = self.engine_obj.engine_props['snapshot_step']
        if self.engine_obj.engine_props['save_snapshots'] and (arrival_count + 1) % snapshot_step == 0:
            self.engine_obj.stats_obj.update_snapshot(net_spec_dict=self.engine_obj.net_spec_dict,
                                                      req_num=arrival_count + 1)

    def step(self, action: list):
        self._update_helper_obj(action=action)
        self.helper_obj.allocate(route_obj=self.route_obj)
        reqs_status_dict = self.engine_obj.reqs_status_dict

        req_id = self.ai_props['arrival_list'][self.ai_props['arrival_count']]['req_id']
        if req_id in reqs_status_dict:
            was_allocated = True
        else:
            was_allocated = False
        self._update_snapshots()

        # TODO: Change algorithm to use props instead of passing every time
        reward = self.helper_obj.calculate_reward(was_allocated=was_allocated, algorithm=self.algorithm)
        self.ai_props['arrival_count'] += 1
        terminated = self._check_terminated()
        new_obs = self._get_obs()
        truncated = False
        info = self._get_info()

        return new_obs, reward, terminated, truncated, info

    @staticmethod
    def _get_info():
        return dict()

    # TODO: This will change but better to wait for Q-Learning integration
    def _get_obs(self):
        # Used when we reach a reset after a simulation has finished (reset automatically called by gymnasium, use
        # placeholder variable)
        if self.ai_props['arrival_count'] == self.engine_obj.engine_props['num_requests']:
            curr_req = self.ai_props['arrival_list'][self.ai_props['arrival_count'] - 1]
        else:
            curr_req = self.ai_props['arrival_list'][self.ai_props['arrival_count']]

        self.ai_props['mock_sdn_dict'] = self.helper_obj.update_mock_sdn(curr_req=curr_req)
        self.route_obj.sdn_props = self.ai_props['mock_sdn_dict']
        self.route_obj.get_route()

        paths_matrix = self.route_obj.route_props['paths_list']
        spectrum_obs = self.helper_obj.get_spectrum(paths_matrix=paths_matrix)

        encode_bw_list = np.zeros((3,))
        if len(self.ai_props['bandwidth_list']) != 0:
            bandwidth_index = self.ai_props['bandwidth_list'].index(curr_req['bandwidth'])
            encode_bw_list[bandwidth_index] = 1

        obs_dict = {
            'source': int(curr_req['source']),
            'destination': int(curr_req['destination']),
            'bandwidth': encode_bw_list,
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

                self.ai_props['arrival_list'].append(self.engine_obj.reqs_dict[req_time])
            else:
                if req_time > self.max_depart:
                    self.max_depart = req_time
                if req_time < self.min_depart:
                    self.min_depart = req_time

                self.ai_props['depart_list'].append(self.engine_obj.reqs_dict[req_time])

    def _create_input(self):
        base_fp = os.path.join('data')
        self.sim_dict['s1']['thread_num'] = 's1'
        get_start_time(sim_dict=self.sim_dict)
        file_name = "sim_input_s1.json"

        self.engine_obj = Engine(engine_props=self.sim_dict['s1'])
        self.route_obj = Routing(engine_props=self.engine_obj.engine_props, sdn_props=self.ai_props['mock_sdn_dict'])
        self.sim_dict['s1'] = create_input(base_fp=base_fp, engine_props=self.sim_dict['s1'])

        if self.options['save_sim']:
            save_input(base_fp=base_fp, properties=self.sim_dict['s1'], file_name=file_name,
                       data_dict=self.sim_dict['s1'])

    def setup(self):
        """
        Sets up this class.
        """
        args_obj = parse_args()
        config_path = os.path.join('ini', 'run_ini', 'config.ini')
        self.sim_dict = read_config(args_obj=args_obj, config_path=config_path)

        self.ai_props['k_paths'] = self.sim_dict['s1']['k_paths']
        self.ai_props['cores_per_link'] = self.sim_dict['s1']['cores_per_link']
        self.ai_props['spectral_slots'] = self.sim_dict['s1']['spectral_slots']

        self._create_input()
        start_arr_rate = float(self.sim_dict['s1']['arrival_rate']['start'])
        self.engine_obj.engine_props['erlang'] = start_arr_rate / self.sim_dict['s1']['holding_time']
        self.engine_obj.engine_props['arrival_rate'] = start_arr_rate * self.sim_dict['s1']['cores_per_link']

    def reset(self, seed: int = None, options: dict = None):  # pylint: disable=arguments-differ
        # Default options
        if options is None:
            self.options = {'save_sim': True}
        else:
            self.options = options

        super().reset(seed=seed)
        self.ai_props = copy.deepcopy(empty_dqn_props)
        self.setup()
        self.ai_props['arrival_count'] = 0
        self.engine_obj.init_iter(iteration=self.iteration)
        self.engine_obj.create_topology()
        self.ai_props['num_nodes'] = len(self.engine_obj.topology.nodes)

        self.helper_obj.ai_props = self.ai_props
        self.helper_obj.engine_obj = self.engine_obj
        self.helper_obj.route_obj = self.route_obj

        if seed is None:
            seed = self.iteration
        self._reset_reqs_dict(seed=seed)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info


# TODO: Have multiple iterations save to one file for predictions
if __name__ == '__main__':
    env = SimEnv(algorithm='ppo')

    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=20000, log_interval=1)

    # model.save('./logs/best_ppo_model.zip')
    # model = DQN.load('./logs/dqn/best_model.zip', env=env)
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
    # # TODO: Save this to a file or plot
    # print(episode_reward / max_episodes)
    # obs, info = env.reset()
