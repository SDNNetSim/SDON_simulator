import os
from datetime import datetime

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

from config_scripts.parse_args import parse_args
from config_scripts.setup_config import read_config
from sim_scripts.engine import Engine
from sim_scripts.routing import Routing
from helper_scripts.setup_helpers import create_input, save_input


# TODO: Threading not supported, will only use s1
# TODO: Slicing is not supported
# TODO: Write tests for these methods
# TODO: Update standards and guidelines document

class DQNEnv(gym.Env):
    metadata = None

    def __init__(self, render_mode: str = None):
        super().__init__()

        self.dqn_sim_dict = None
        self.engine_obj = None
        self.route_obj = None
        # TODO: Make this better
        self.mock_sdn = dict()
        self.setup()
        self.engine_props = self.engine_obj.engine_props

        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(self.engine_props['cores_per_link'],
                                                   self.engine_props['spectral_slots']),
                                            dtype=np.float64)
        self.action_space = spaces.Discrete(self.engine_props['cores_per_link'] *
                                            self.engine_props['spectral_slots'])

        # TODO: Ensure each of these are updated correctly in step
        #   - Also, how to update iteration (based on gym env maybe)
        #   - After, integration with saving properly
        self.iteration = 0
        self.req_num = 1
        self.curr_time = None

    def _check_terminated(self):
        # TODO: Check to make sure this is correct (or if it's minus one)
        if self.req_num == ((self.engine_props['num_requests'] * 2) + 1):
            terminated = True
            base_fp = os.path.join('..', 'data')
            self.engine_obj.end_iter(base_fp=base_fp, iteration=self.iteration, print_flag=False, ai_flag=True)
            self.iteration += 1
        else:
            terminated = False

        return terminated

    @staticmethod
    def _calculate_reward(was_allocated: bool):
        if was_allocated:
            reward = 1
        else:
            reward = -1

        return reward

    def _update_reqs_status(self, was_routed: bool, mod_format: str = None, path_list: list = None):
        self.engine_obj.reqs_status_dict.update({self.engine_obj.reqs_dict[self.curr_time]['req_id']: {
            "mod_format": mod_format,
            "path": path_list,
            "is_sliced": False,
            "was_routed": was_routed,
        }})

    # TODO: Move these params to props or something
    # TODO: Neaten this method up
    def _update_net_spec_dict(self, path_list: list, core_num: int, start_slot: int, end_slot: int):
        for link_tuple in zip(path_list, path_list[1:]):
            self.engine_obj.net_spec_dict[(link_tuple[0], link_tuple[1])]['cores_matrix'][core_num][
            start_slot:end_slot] = self.req_num
            self.engine_obj.net_spec_dict[(link_tuple[1], link_tuple[0])]['cores_matrix'][core_num][
            start_slot:end_slot] = self.req_num

            self.engine_obj.net_spec_dict[(link_tuple[0], link_tuple[1])]['cores_matrix'][core_num][
                end_slot] = self.req_num * -1
            self.engine_obj.net_spec_dict[(link_tuple[1], link_tuple[0])]['cores_matrix'][core_num][
                end_slot] = self.req_num * -1

    def _check_is_free(self, path_list: list, core_num: int, start_slot: int, end_slot: int):
        is_free = True
        link_dict = None
        rev_link_dict = None
        for link_tuple in zip(path_list, path_list[1:]):
            link_dict = self.engine_obj.net_spec_dict[(link_tuple[0], link_tuple[1])]
            rev_link_dict = self.engine_obj.net_spec_dict[(link_tuple[1], link_tuple[0])]

            tmp_set = set(link_dict['cores_matrix'][core_num][start_slot:end_slot + 1])
            rev_tmp_set = set(rev_link_dict['cores_matrix'][core_num][start_slot:end_slot + 1])

            if tmp_set != {0.0} or rev_tmp_set != {0.0}:
                is_free = False

        return is_free, link_dict, rev_link_dict

    def release(self):
        arrival_id = self.engine_obj.reqs_dict[self.curr_time]['req_id']
        if self.engine_obj.reqs_status_dict[arrival_id]['was_routed']:
            path_list = self.engine_obj.reqs_status_dict[arrival_id]['path']
            for source, dest in zip(path_list, path_list[1:]):
                for core_num in range(self.engine_props['cores_per_link']):
                    core_arr = self.engine_obj.net_spec_dict[(source, dest)]['cores_matrix'][core_num]
                    req_id_arr = np.where(core_arr == arrival_id)
                    gb_arr = np.where(core_arr == (arrival_id * -1))

                    for req_index, gb_index in zip(req_id_arr, gb_arr):
                        self.engine_obj.net_spec_dict[(source, dest)]['cores_matrix'][core_num][req_index] = 0
                        self.engine_obj.net_spec_dict[(dest, source)]['cores_matrix'][core_num][req_index] = 0

                        self.engine_obj.net_spec_dict[(source, dest)]['cores_matrix'][core_num][gb_index] = 0
                        self.engine_obj.net_spec_dict[(dest, source)]['cores_matrix'][core_num][gb_index] = 0
        # Request was blocked
        else:
            pass

    # TODO: Check to see if this works with multiple paths
    # TODO: Clean this function up
    def allocate(self, core_num: int, start_slot: int):
        was_allocated = True
        self.mock_sdn['was_routed'] = True
        for path_index, path_list in enumerate(self.route_obj.route_props['paths_list']):
            path_len = self.route_obj.route_props['weights_list'][path_index]
            mod_format = self.route_obj.route_props['mod_formats_list'][path_index][0]
            self.mock_sdn['path_list'] = path_list
            if not mod_format:
                self.mock_sdn['was_routed'] = False
                was_allocated = False
                continue

            bandwidth = self.engine_obj.reqs_dict[self.curr_time]['bandwidth']
            bandwidth_dict = self.engine_props['mod_per_bw'][bandwidth]
            end_slot = start_slot + bandwidth_dict[mod_format]['slots_needed']
            if end_slot >= self.engine_props['spectral_slots']:
                self.mock_sdn['was_routed'] = False
                was_allocated = False
                continue

            is_free, link_dict, rev_link_dict = self._check_is_free(path_list=path_list, core_num=core_num,
                                                                    start_slot=start_slot,
                                                                    end_slot=end_slot)

            if is_free:
                self._update_net_spec_dict(path_list=path_list, core_num=core_num,
                                           start_slot=start_slot,
                                           end_slot=end_slot)
                self._update_reqs_status(path_list=path_list, was_routed=True, mod_format=mod_format)

                self.mock_sdn['bandwidth_list'].append(bandwidth)
                self.mock_sdn['modulation_list'].append(mod_format)
                self.mock_sdn['core_list'].append(core_num)
                self.mock_sdn['path_weight'] = path_len
                self.mock_sdn['spectrum_dict']['modulation'] = mod_format

                self.mock_sdn['was_routed'] = True
                was_allocated = True
                return was_allocated
            else:
                was_allocated = False
                self.mock_sdn['was_routed'] = False

        self._update_reqs_status(was_routed=False)
        return was_allocated

    # TODO: What if second request is a departure?
    def step(self, action: int):
        # TODO: Skeptical about this
        core_num = action // self.engine_props['spectral_slots']
        start_slot = action % self.engine_props['spectral_slots']

        was_allocated = self.allocate(core_num=core_num, start_slot=start_slot)
        if self.mock_sdn['was_routed'] and len(self.mock_sdn['modulation_list']) == 0:
            print('Here')

        self.engine_obj.update_arrival_params(curr_time=self.curr_time, ai_flag=True, mock_sdn=self.mock_sdn)

        reward = self._calculate_reward(was_allocated=was_allocated)

        while True:
            self.curr_time = list(self.engine_obj.reqs_dict.keys())[self.req_num - 1]
            if self.engine_obj.reqs_dict[self.curr_time]['request_type'] == 'release':
                self.release()
                self.req_num += 1
                terminated = self._check_terminated()
                if terminated:
                    break
            else:
                self.req_num += 1
                terminated = self._check_terminated()
                break

        new_obs = self._get_obs()
        truncated = False
        info = self._get_info()

        return new_obs, reward, terminated, truncated, info

    @staticmethod
    def _get_info():
        return dict()

    @staticmethod
    def combine_and_one_hot(arr1, arr2):
        if len(arr1) != len(arr2):
            raise ValueError("Arrays must have the same length.")

        one_hot_arr1 = (arr1 != 0).astype(int)
        one_hot_arr2 = (arr2 != 0).astype(int)

        result = one_hot_arr1 | one_hot_arr2
        return result

    def _get_spectrum(self, paths_matrix: list):
        spectrum_matrix = np.zeros((self.engine_props['cores_per_link'], self.engine_props['spectral_slots']))
        for paths_list in paths_matrix:
            for link_tuple in zip(paths_list, paths_list[1:]):
                link_dict = self.engine_obj.net_spec_dict[(link_tuple[0], link_tuple[1])]
                rev_link_dict = self.engine_obj.net_spec_dict[(link_tuple[1], link_tuple[0])]

                if link_dict != rev_link_dict:
                    raise ValueError('Link is not bi-directionally equal.')

                for core_index, core_arr in enumerate(link_dict['cores_matrix']):
                    spectrum_matrix[core_index] = self.combine_and_one_hot(arr1=spectrum_matrix[core_index],
                                                                           arr2=core_arr)

        return spectrum_matrix

    def _update_mock_sdn(self, curr_req: dict):
        self.mock_sdn = {
            'source': curr_req['source'],
            'destination': curr_req['destination'],
            'bandwidth': curr_req['bandwidth'],
            'net_spec_dict': self.engine_obj.net_spec_dict,
            'topology': self.engine_obj.topology,
            'mod_formats': curr_req['mod_formats'],
            # TODO: Route time, num trans, and block reason static temporarily
            'num_trans': 1.0,
            'route_time': 0.0,
            'block_reason': 'congestion',
            'stat_key_list': ['modulation_list', 'xt_list', 'core_list'],
            'modulation_list': list(),
            'xt_list': list(),
            'is_sliced': False,
            'core_list': list(),
            'bandwidth_list': list(),
            'path_weight': list(),
            'spectrum_dict': {'modulation': None}
        }

    def _get_obs(self):
        curr_req = self.engine_obj.reqs_dict[self.curr_time]
        self._update_mock_sdn(curr_req=curr_req)

        self.route_obj.sdn_props = self.mock_sdn
        self.route_obj.get_route()

        paths_matrix = self.route_obj.route_props['paths_list']
        spectrum_obs = self._get_spectrum(paths_matrix=paths_matrix)
        return spectrum_obs

    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed)

        self.req_num = 1
        self.engine_obj.init_iter(iteration=self.iteration)
        self.engine_obj.create_topology()

        if seed is None:
            seed = self.iteration
        self.engine_obj.generate_requests(seed=seed)
        self.curr_time = list(self.engine_obj.reqs_dict.keys())[self.req_num - 1]

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _get_start_time(self):
        sim_start = datetime.now().strftime("%m%d_%H_%M_%S_%f")
        self.dqn_sim_dict['s1']['date'] = sim_start.split('_')[0]
        tmp_list = sim_start.split('_')

        time_string = f'{tmp_list[1]}_{tmp_list[2]}_{tmp_list[3]}_{tmp_list[4]}'
        self.dqn_sim_dict['s1']['sim_start'] = time_string

    def setup(self):
        args_obj = parse_args()
        config_path = os.path.join('..', 'ini', 'run_ini', 'config.ini')
        self.dqn_sim_dict = read_config(args_obj=args_obj, config_path=config_path)

        base_fp = os.path.join('..', 'data')
        self.dqn_sim_dict['s1']['thread_num'] = 's1'
        self._get_start_time()
        self.dqn_sim_dict['s1'] = create_input(base_fp=base_fp, engine_props=self.dqn_sim_dict['s1'])
        file_name = f"sim_input_s1.json"
        save_input(base_fp=base_fp, properties=self.dqn_sim_dict['s1'], file_name=file_name,
                   data_dict=self.dqn_sim_dict['s1'])

        self.engine_obj = Engine(engine_props=self.dqn_sim_dict['s1'])
        self.route_obj = Routing(engine_props=self.engine_obj.engine_props, sdn_props=self.mock_sdn)

        # TODO: Save for initial arrival rate
        # TODO: Figure out how to loop through inter arrival times
        #   - Or, should we train on one type of traffic
        #   - Maybe have a loop to create a list of arrival rates and then handle it by index
        #   - This might get confusing since we can't just return back here...
        # TODO: Utilize options if you can
        start_arr_rate = float(self.dqn_sim_dict['s1']['arrival_rate']['start'])
        self.engine_obj.engine_props['erlang'] = start_arr_rate / self.dqn_sim_dict['s1']['holding_time']
        self.engine_obj.engine_props['arrival_rate'] = start_arr_rate * self.dqn_sim_dict['s1']['cores_per_link']


# TODO: Handle saving the model
if __name__ == '__main__':
    env = DQNEnv()
    # check_env(env)

    model = DQN("MlpPolicy", env, verbose=1)
    # TODO: Manually use iters, time steps should be num requests times iters
    model.learn(total_timesteps=10, log_interval=5)
    # model.save("dqn_model")

    # del model  # remove to demonstrate saving and loading

    # model = DQN.load("dqn_model")

    obs, info = env.reset()
    while True:
        curr_action, _states = model.predict(obs, deterministic=True)

        obs, curr_reward, is_terminated, is_truncated, curr_info = env.step(curr_action)
        if is_terminated or is_truncated:
            obs, info = env.reset()
