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
from helper_scripts.rl_helpers import RLHelpers
from helper_scripts.callback_helpers import GetModelParams
from helper_scripts.sim_helpers import get_start_time, find_path_len, get_path_mod, find_path_cong
from arg_scripts.rl_args import empty_drl_props, empty_q_props, empty_rl_props
from helper_scripts.multi_agent_helpers import PathAgent, CoreAgent, SpectrumAgent


# TODO: Account for command line input
#   - Run command line input in this script
# TODO: AI props should be RL props
# TODO: Only support for s1
# TODO: Plan is to run and debug the path agent today with new formulations
#   - Run overnight the path agent considering cong. and frag.
#   - Not sure how my formulation will be for this just yet
class SimEnv(gym.Env):  # pylint: disable=abstract-method
    """
    Simulates a deep q-learning environment with stable baselines3 integration.
    """
    metadata = dict()

    def __init__(self, render_mode: str = None, custom_callback: object = None, **kwargs):
        super().__init__()

        # TODO: Double check constructors used, probably won't need many now
        self.rl_props = copy.deepcopy(empty_rl_props)
        self.rl_props['super_channel_space'] = None
        self.q_props = copy.deepcopy(empty_q_props)
        self.drl_props = copy.deepcopy(empty_drl_props)

        self.sim_dict = dict()

        self.iteration = 0
        self.options = None
        self.optimize = None
        self.callback = custom_callback
        self.render_mode = render_mode

        self.engine_obj = None
        self.route_obj = None
        self.helper_obj = RLHelpers(ai_props=self.rl_props, engine_obj=self.engine_obj, route_obj=self.route_obj,
                                    q_props=self.q_props, drl_props=self.drl_props)

        # TODO: Not sure how I'll init constructor vars just yet
        self.multi_agent_obj = {
            'path_agent': PathAgent(),
            'core_agent': CoreAgent(),
            'spectrum_agent': SpectrumAgent(),
        }

        self.path_algorithm = None
        self.core_algorithm = None
        self.spectrum_algorithm = None

        self.paths_obj = None
        # Used to determine level of congestion, fragmentation, etc. for the q-learning algorithm
        self.level_index = None

        # Used to get config variables into the observation space
        self.reset(options={'save_sim': False})
        # TODO: Change for multi-agent (DQN, PPO, A2C)
        self.observation_space = self.helper_obj.get_obs_space()
        # TODO: Change for multi-agent (DQN, PPO, A2C)
        self.action_space = self.helper_obj.get_action_space()

    def _check_terminated(self):
        if self.rl_props['arrival_count'] == (self.engine_obj.engine_props['num_requests']):
            terminated = True
            base_fp = os.path.join('data')
            # TODO: Update amount
            if self.path_algorithm == 'q_learning':
                amount = 0
                # self.helper_obj.decay_epsilon(amount=amount, iteration=self.iteration)
            self.engine_obj.end_iter(iteration=self.iteration, print_flag=False, ai_flag=True, base_fp=base_fp)
            self.iteration += 1
        else:
            terminated = False

        return terminated

    def _error_check_actions(self):
        if self.helper_obj.path_index < 0 or self.helper_obj.path_index > (self.rl_props['k_paths'] - 1):
            raise ValueError(f'Path index out of range: {self.helper_obj.path_index}')
        if self.helper_obj.core_num < 0 or self.helper_obj.core_num > (
                self.rl_props['cores_per_link'] - 1):
            raise ValueError(f'Core index out of range: {self.helper_obj.core_num}')

    def _update_helper_obj(self, action: list):
        # TODO: Want spectrum assignment to do this
        self.helper_obj.path_index = self.rl_props['path_index']
        # TODO: No more picking a core for now
        # self.helper_obj.core_num = self.rl_props['core_index']

        # TODO: First or best fit, depends on configuration file
        # if self.path_algorithm == 'q_learning':
        #     raise NotImplementedError
        # else:
        # TODO: Change this to pick a super channel
        #   - Only valid super-channels
        #   - If None, block
        #       - Reward will be zero in this case
        # self.helper_obj.super_channel = action
        # self._error_check_actions()

        if self.path_algorithm == 'q_learning':
            self.helper_obj.q_props = self.q_props
        else:
            self.helper_obj.drl_props = self.drl_props

        self.helper_obj.ai_props = self.rl_props
        self.helper_obj.engine_obj = self.engine_obj
        self.helper_obj.handle_releases()

    def step(self, action: list):
        self._update_helper_obj(action=action)
        # TODO: Make sure route object has the selected path for q-learning
        self.helper_obj.allocate(route_obj=self.route_obj)
        reqs_status_dict = self.engine_obj.reqs_status_dict
        req_id = self.rl_props['arrival_list'][self.rl_props['arrival_count']]['req_id']

        if req_id in reqs_status_dict:
            was_allocated = True
        else:
            was_allocated = False
        self._update_snapshots()

        drl_reward = self.helper_obj.calculate_drl_reward(was_allocated=was_allocated)

        if self.path_algorithm == 'q_learning':
            self._update_routes_matrix(was_routed=was_allocated)
            # self._update_cores_matrix(was_routed=was_allocated)
        self.rl_props['arrival_count'] += 1
        terminated = self._check_terminated()
        new_obs = self._get_obs()
        truncated = False
        info = self._get_info()

        return new_obs, drl_reward, terminated, truncated, info

    @staticmethod
    def _get_info():
        return dict()

    def _get_obs(self):
        # Used when we reach a reset after a simulation has finished (reset automatically called by gymnasium, use
        # placeholder variable)
        if self.rl_props['arrival_count'] == self.engine_obj.engine_props['num_requests']:
            curr_req = self.rl_props['arrival_list'][self.rl_props['arrival_count'] - 1]
        else:
            curr_req = self.rl_props['arrival_list'][self.rl_props['arrival_count']]

        self.rl_props['source'] = int(curr_req['source'])
        self.rl_props['destination'] = int(curr_req['destination'])
        self.rl_props['mock_sdn_dict'] = self.helper_obj.update_mock_sdn(curr_req=curr_req)
        if self.path_algorithm == 'q_learning':
            self.get_route()
            # TODO: Core for now will be first-fit in spectrum assignment only
            # self.get_core()
            path_len = find_path_len(path_list=self.rl_props['paths_list'][self.rl_props['path_index']],
                                     topology=self.engine_obj.topology)
            path_mod = get_path_mod(mods_dict=curr_req['mod_formats'], path_len=path_len)
        # TODO: At the moment only works for SPF-FF
        else:
            self.route_obj.sdn_props = self.rl_props['mock_sdn_dict']
            self.route_obj.get_route()
            self.rl_props['paths_list'] = self.route_obj.route_props['paths_list']
            self.rl_props['path_index'] = 0
            self.rl_props['core_index'] = 0
            path_mod = self.route_obj.route_props['mod_formats_list'][0][0]

        if path_mod is not False:
            slots_needed = curr_req['mod_formats'][path_mod]['slots_needed']
        else:
            slots_needed = 0
        # super_channels = self.helper_obj.get_super_channels(slots_needed=slots_needed,
        #                                                     num_channels=self.super_channel_space)

        source_obs = np.zeros(self.rl_props['num_nodes'])
        source_obs[self.rl_props['source']] = 1.0
        dest_obs = np.zeros(self.rl_props['num_nodes'])
        dest_obs[self.rl_props['destination']] = 1.0

        obs_dict = {
            'slots_needed': slots_needed,
            'source': source_obs,
            'destination': dest_obs,
            # TODO: Change
            'super_channels': 0,
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

                self.rl_props['arrival_list'].append(self.engine_obj.reqs_dict[req_time])
            else:
                if req_time > self.max_depart:
                    self.max_depart = req_time
                if req_time < self.min_depart:
                    self.min_depart = req_time

                self.rl_props['depart_list'].append(self.engine_obj.reqs_dict[req_time])

    def _create_input(self):
        base_fp = os.path.join('data')
        self.sim_dict['s1']['thread_num'] = 's1'
        get_start_time(sim_dict=self.sim_dict)
        file_name = "sim_input_s1.json"

        self.engine_obj = Engine(engine_props=self.sim_dict['s1'])
        self.route_obj = Routing(engine_props=self.engine_obj.engine_props,
                                 sdn_props=self.rl_props['mock_sdn_dict'])
        self.sim_dict['s1'] = create_input(base_fp=base_fp, engine_props=self.sim_dict['s1'])
        save_input(base_fp=base_fp, properties=self.sim_dict['s1'], file_name=file_name,
                   data_dict=self.sim_dict['s1'])

    def setup(self):
        """
        Sets up this class.
        """
        # args_obj = parse_args()
        # config_path = os.path.join('ini', 'run_ini', 'config.ini')
        # self.sim_dict = read_config(args_obj=args_obj, config_path=config_path)

        # Instead of args obj
        self.optimize = self.sim_dict['optimize']
        self.rl_props['k_paths'] = self.sim_dict['s1']['k_paths']
        self.rl_props['cores_per_link'] = self.sim_dict['s1']['cores_per_link']
        self.rl_props['spectral_slots'] = self.sim_dict['s1']['spectral_slots']

        self._create_input()

        # TODO: These will be changed
        self.path_algorithm = self.sim_dict['s1']['path_algorithm']
        self.core_algorithm = 'first_fit'
        self.spectrum_algorithm = 'first_fit'

        start_arr_rate = float(self.sim_dict['s1']['arrival_rate']['start'])
        self.engine_obj.engine_props['erlang'] = start_arr_rate / self.sim_dict['s1']['holding_time']
        self.engine_obj.engine_props['arrival_rate'] = start_arr_rate * self.sim_dict['s1']['cores_per_link']

    def reset(self, seed: int = None, options: dict = None):  # pylint: disable=arguments-differ
        super().reset(seed=seed)
        self.rl_props['arrival_list'] = list()
        self.rl_props['depart_list'] = list()

        if self.optimize or self.optimize is None:
            # TODO: These will have to be modified
            self.rl_props['q_props'] = copy.deepcopy(empty_q_props)
            self.rl_props['drl_props'] = copy.deepcopy(empty_drl_props)
            self.iteration = 0
            self.setup()

        self.rl_props['arrival_count'] = 0
        self.engine_obj.init_iter(iteration=self.iteration)
        self.engine_obj.create_topology()
        self.rl_props['num_nodes'] = len(self.engine_obj.topology.nodes)

        # TODO: Needs to be generalized
        if self.path_algorithm == 'q_learning' and self.iteration == 0:
            self.setup_q_env()
            self.helper_obj.q_props = self.q_props
        else:
            self.helper_obj.q_props = self.q_props
            # self.helper_obj.drl_props = self.drl_props

        self.helper_obj.ai_props = self.rl_props
        self.helper_obj.engine_obj = self.engine_obj
        self.helper_obj.route_obj = self.route_obj

        if seed is None:
            # TODO: Change
            # seed = self.iteration
            seed = 0

        self._reset_reqs_dict(seed=seed)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info


def _run_iter():
    raise NotImplementedError


def _run_testing(env: object, sim_dict: dict):
    # model = DQN.load('./logs/DQN/best_model.zip', env=env)
    raise NotImplementedError


def _get_model(algorithm: str, device: str, env: object):
    model = None
    if algorithm == 'dqn':
        model = None
    elif algorithm == 'ppo':
        model = None
    elif algorithm == 'a2c':
        model = None

    return model


def _print_train_info(sim_dict: dict):
    if sim_dict['path_algorithm'] == 'q_learning':
        print(f'Beginning training process for the PATH AGENT using the '
              f'{sim_dict["path_algorithm"].title()} algorithm.')
    elif sim_dict['core_algorithm'] == 'q_learning':
        print(f'Beginning training process for the CORE AGENT using the '
              f'{sim_dict["core_algorithm"].title()} algorithm.')
    elif sim_dict['spectrum_algorithm'] in ('dqn', 'ppo', 'a2c'):
        print(f'Beginning training process for the SPECTRUM AGENT using the '
              f'{sim_dict["spectrum_algorithm"].title()} algorithm.')
    else:
        raise ValueError(f'Invalid algorithm received or all algorithms are not reinforcement learning. '
                         f'Expected: q_learning, dqn, ppo, a2c, Got: {sim_dict["path_algorithm"]}, '
                         f'{sim_dict["core_algorithm"]}, {sim_dict["spectrum_algorithm"]}')


def _run_training(env: object, sim_dict: dict):
    _print_train_info(sim_dict=sim_dict)
    if sim_dict['path_algorithm'] == 'q_learning' or sim_dict['core_algorithm'] == 'q_learning':
        raise NotImplementedError
    elif sim_dict['spectrum_algorithm'] in ('dqn', 'ppo', 'a2c'):
        model = _get_model(algorithm=sim_dict['spectrum_algorithm'], device=sim_dict['device'], env=env)
        # TODO: This should come from the yml file actually (total time steps and print step)?
        model.learn(total_timesteps=sim_dict['num_requests'], log_interval=sim_dict['print_step'],
                    callback=sim_dict['callback'])
        # TODO: Save model
        # model.save('./logs/best_PPO_model.zip')
    else:
        raise ValueError(f'Invalid algorithm received or all algorithms are not reinforcement learning. '
                         f'Expected: q_learning, dqn, ppo, a2c, Got: {sim_dict["path_algorithm"]}, '
                         f'{sim_dict["core_algorithm"]}, {sim_dict["spectrum_algorithm"]}')


def _setup_rl_sim():
    args_obj = parse_args()
    config_path = os.path.join('ini', 'run_ini', 'config.ini')
    sim_dict = read_config(args_obj=args_obj, config_path=config_path)

    return sim_dict


# TODO: To run RLZoo by command line here, we only need to run this script!
# TODO: Maybe also run register env and other important things similar to that
def run_rl_sim():
    callback = GetModelParams()
    env = SimEnv(render_mode=None, custom_callback=callback)
    env.sim_dict = _setup_rl_sim()
    env.sim_dict['callback'] = callback

    if env.sim_dict['is_training']:
        _run_training(env=env, sim_dict=env.sim_dict)
    else:
        _run_testing(env=env, sim_dict=env.sim_dict)

    obs, info = env.reset()
    # TODO: Reward should be saved in their individual files, not needed here
    episode_reward = 0
    # TODO: Taken from max iterations not here
    max_episodes = 100
    num_episodes = 0
    time_steps = 0
    while True:
        # curr_action, _states = model.predict(obs)

        obs, curr_reward, is_terminated, is_truncated, curr_info = env.step([0])
        episode_reward += curr_reward

        time_steps += 1

        # print(f'{time_steps} Time steps completed.')

        if num_episodes >= max_episodes:
            break
        if is_terminated or is_truncated:
            obs, info = env.reset()
            num_episodes += 1
            print(f'{num_episodes} episodes completed.')
    #
    # # TODO: Save this to a file or plot
    print(episode_reward / max_episodes)
    # obs, info = env.reset()


if __name__ == '__main__':
    run_rl_sim()
