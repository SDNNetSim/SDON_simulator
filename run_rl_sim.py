import os
import copy
import torch.nn as nn

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
from helper_scripts.sim_helpers import get_start_time, find_path_len, get_path_mod, parse_yaml_file
from arg_scripts.rl_args import empty_rl_props
from helper_scripts.multi_agent_helpers import PathAgent, CoreAgent, SpectrumAgent


# TODO: Re-order functions
# TODO: Check if props needs to be reset
# TODO: Check if props updated in other objects
# TODO: RLZoo handled via command line
# TODO: Goal is for everything to be ran here
class SimEnv(gym.Env):  # pylint: disable=abstract-method
    metadata = dict()

    def __init__(self, render_mode: str = None, custom_callback: object = None, sim_dict: dict = None, **kwargs):
        super().__init__()

        self.rl_props = copy.deepcopy(empty_rl_props)
        self.sim_dict = sim_dict['s1']
        self.rl_props['super_channel_space'] = self.sim_dict['super_channel_space']

        self.iteration = 0
        self.options = None
        self.optimize = None
        self.callback = custom_callback
        self.render_mode = render_mode

        self.engine_obj = None
        self.route_obj = None
        self.rl_help_obj = RLHelpers(rl_props=self.rl_props, engine_obj=self.engine_obj, route_obj=self.route_obj)

        self.path_agent = PathAgent(path_algorithm=self.sim_dict['path_algorithm'], rl_props=self.rl_props,
                                    rl_help_obj=self.rl_help_obj)
        self.core_agent = CoreAgent(core_algorithm=self.sim_dict['core_algorithm'], rl_props=self.rl_props,
                                    rl_help_obj=self.rl_help_obj)
        self.spectrum_agent = SpectrumAgent(spectrum_algorithm=self.sim_dict['spectrum_algorithm'],
                                            rl_props=self.rl_props)

        self.modified_props = None
        # Used to get config variables into the observation space
        self.reset(options={'save_sim': False})
        self.observation_space = self.spectrum_agent.get_obs_space()
        self.action_space = self.spectrum_agent.get_action_space()

    def _check_terminated(self):
        if self.rl_props['arrival_count'] == (self.engine_obj.engine_props['num_requests']):
            terminated = True
            base_fp = os.path.join('data')
            if self.sim_dict['path_algorithm'] == 'q_learning' and self.sim_dict['is_training']:
                self.path_agent.end_iter()
            elif self.sim_dict['core_algorithm'] == 'q_learning' and self.sim_dict['is_training']:
                self.core_agent.end_iter()
            self.engine_obj.end_iter(iteration=self.iteration, print_flag=False, ai_flag=True, base_fp=base_fp)
            self.iteration += 1
        else:
            terminated = False

        return terminated

    def _update_helper_obj(self, action: list, bandwidth):
        self.rl_help_obj.path_index = self.rl_props['path_index']
        self.rl_help_obj.core_num = self.rl_props['core_index']

        if self.sim_dict['spectrum_algorithm'] in ('dqn', 'ppo', 'a2c'):
            self.rl_help_obj.rl_props['forced_index'] = action
        else:
            self.rl_help_obj.rl_props['forced_index'] = None

        self.rl_help_obj.ai_props = self.rl_props
        self.rl_help_obj.engine_obj = self.engine_obj
        self.rl_help_obj.handle_releases()
        self.rl_help_obj.update_route_props(chosen_path=self.rl_props['chosen_path'], bandwidth=bandwidth)

    def _handle_test_train_step(self, was_allocated: bool):
        if self.sim_dict['is_training']:
            if self.sim_dict['path_algorithm'] == 'q_learning':
                self.path_agent.update(was_allocated=was_allocated, net_spec_dict=self.engine_obj.net_spec_dict,
                                       iteration=self.iteration)
            elif self.sim_dict['core_algorithm'] == 'q_learning':
                self.core_agent.update(was_allocated=was_allocated, net_spec_dict=self.engine_obj.net_spec_dict,
                                       iteration=self.iteration)
        else:
            raise NotImplementedError

    def step(self, action: list):
        req_info_dict = self.rl_props['arrival_list'][self.rl_props['arrival_count']]
        req_id = req_info_dict['req_id']
        bandwidth = req_info_dict['bandwidth']
        self._update_helper_obj(action=action, bandwidth=bandwidth)
        self.rl_help_obj.allocate()
        reqs_status_dict = self.engine_obj.reqs_status_dict

        if req_id in reqs_status_dict:
            was_allocated = True
        else:
            was_allocated = False

        self._handle_test_train_step(was_allocated=was_allocated)
        self.rl_help_obj.update_snapshots()
        drl_reward = self.spectrum_agent.get_reward(was_allocated=was_allocated)

        self.rl_props['arrival_count'] += 1
        terminated = self._check_terminated()
        new_obs = self._get_obs()
        truncated = False
        info = self._get_info()

        return new_obs, drl_reward, terminated, truncated, info

    @staticmethod
    def _get_info():
        return dict()

    def _handle_path_train(self):
        self.path_agent.get_route()
        self.rl_help_obj.rl_props['chosen_path'] = [self.rl_props['chosen_path']]
        self.route_obj.route_props['paths_list'] = self.rl_help_obj.rl_props['chosen_path']
        self.rl_props['core_index'] = None
        self.rl_props['forced_index'] = None

    def _handle_core_train(self):
        self.route_obj.sdn_props = self.rl_props['mock_sdn_dict']
        self.route_obj.engine_props['route_method'] = 'shortest_path'
        self.route_obj.get_route()

        self.rl_props['chosen_path'] = self.route_obj.route_props['paths_list']
        # Always the shortest path
        self.rl_props['path_index'] = 0
        self.rl_props['forced_index'] = None
        self.core_agent.get_core()

    def _handle_spectrum_train(self):
        self.route_obj.sdn_props = self.rl_props['mock_sdn_dict']
        self.route_obj.engine_props['route_method'] = 'shortest_path'
        self.route_obj.get_route()
        self.rl_props['paths_list'] = self.route_obj.route_props['paths_list']
        self.rl_props['chosen_path'] = self.route_obj.route_props['paths_list']
        self.rl_props['path_index'] = 0
        self.rl_props['core_index'] = None

    def _handle_test_train_obs(self, curr_req: dict):
        if self.sim_dict['is_training']:
            if self.sim_dict['path_algorithm'] == 'q_learning':
                self._handle_path_train()
            elif self.sim_dict['core_algorithm'] == 'q_learning':
                self._handle_core_train()
            elif self.sim_dict['spectrum_algorithm'] not in ('first_fit', 'best_fit', ' last_fit'):
                self._handle_spectrum_train()
            else:
                raise NotImplementedError
        # TODO: Load model somewhere else and already have the functions here
        else:
            raise NotImplementedError

        path_len = find_path_len(path_list=self.rl_props['chosen_path'],
                                 topology=self.engine_obj.topology)
        path_mod = get_path_mod(mods_dict=curr_req['mod_formats'], path_len=path_len)

        return path_mod

    def _get_obs(self):
        # Used when we reach a reset after a simulation has finished (reset automatically called by gymnasium, use
        # placeholder variable)
        if self.rl_props['arrival_count'] == self.engine_obj.engine_props['num_requests']:
            curr_req = self.rl_props['arrival_list'][self.rl_props['arrival_count'] - 1]
        else:
            curr_req = self.rl_props['arrival_list'][self.rl_props['arrival_count']]

        self.rl_props['source'] = int(curr_req['source'])
        self.rl_props['destination'] = int(curr_req['destination'])
        self.rl_props['mock_sdn_dict'] = self.rl_help_obj.update_mock_sdn(curr_req=curr_req)

        path_mod = self._handle_test_train_obs(curr_req=curr_req)
        if path_mod is not False:
            slots_needed = curr_req['mod_formats'][path_mod]['slots_needed']
            super_channels, no_penalty = self.rl_help_obj.get_super_channels(slots_needed=slots_needed,
                                                                             num_channels=self.rl_props[
                                                                                 'super_channel_space'])
        # No penalty for DRL agent, mistake not made by it
        else:
            slots_needed = -1
            no_penalty = True
            super_channels = np.array([100.0, 100.0, 100.0])

        self.spectrum_agent.no_penalty = no_penalty
        source_obs = np.zeros(self.rl_props['num_nodes'])
        source_obs[self.rl_props['source']] = 1.0
        dest_obs = np.zeros(self.rl_props['num_nodes'])
        dest_obs[self.rl_props['destination']] = 1.0

        obs_dict = {
            'slots_needed': slots_needed,
            'source': source_obs,
            'destination': dest_obs,
            'super_channels': super_channels,
        }
        return obs_dict

    def _init_envs(self):
        # SB3 will init the environment for us, but not for non-DRL algorithms we've added
        if self.sim_dict['path_algorithm'] == 'q_learning' and self.sim_dict['is_training']:
            self.path_agent.engine_props = self.engine_obj.engine_props
            self.path_agent.setup_env()
        elif self.sim_dict['core_algorithm'] == 'q_learning' and self.sim_dict['is_training']:
            self.core_agent.engine_props = self.engine_obj.engine_props
            self.core_agent.setup_env()

    def _create_input(self):
        base_fp = os.path.join('data')
        self.sim_dict['thread_num'] = 's1'
        # Added only for structure consistency
        get_start_time(sim_dict={'s1': self.sim_dict})
        file_name = "sim_input_s1.json"

        self.engine_obj = Engine(engine_props=self.sim_dict)
        self.route_obj = Routing(engine_props=self.engine_obj.engine_props,
                                 sdn_props=self.rl_props['mock_sdn_dict'])

        self.sim_props = create_input(base_fp=base_fp, engine_props=self.sim_dict)
        self.modified_props = copy.deepcopy(self.sim_props)
        if 'topology' in self.sim_props:
            self.modified_props.pop('topology')
            self.modified_props.pop('callback')

        save_input(base_fp=base_fp, properties=self.modified_props, file_name=file_name,
                   data_dict=self.modified_props)

    # TODO: Options to switch between them
    def _load_models(self):
        self.path_agent.load_model(model_path=self.sim_dict['path_model'])
        self.core_agent.load_model(model_path=self.sim_dict['core_model'])
        self.spectrum_agent.load_model(model_path=self.sim_dict['spectrum_model'])

    def setup(self):
        """
        Sets up this class.
        """
        self.optimize = self.sim_dict['optimize']
        self.rl_props['k_paths'] = self.sim_dict['k_paths']
        self.rl_props['cores_per_link'] = self.sim_dict['cores_per_link']
        self.rl_props['spectral_slots'] = self.sim_dict['spectral_slots']

        self._create_input()

        start_arr_rate = float(self.sim_dict['arrival_rate']['start'])
        self.engine_obj.engine_props['erlang'] = start_arr_rate / self.sim_dict['holding_time']
        self.engine_obj.engine_props['arrival_rate'] = start_arr_rate * self.sim_dict['cores_per_link']

    def _init_props_envs(self):
        self.rl_props['arrival_count'] = 0
        self.engine_obj.init_iter(iteration=self.iteration)
        self.engine_obj.create_topology()
        self.rl_help_obj.topology = self.engine_obj.topology
        self.rl_props['num_nodes'] = len(self.engine_obj.topology.nodes)

        if self.iteration == 0:
            self._init_envs()

        self.rl_help_obj.rl_props = self.rl_props
        self.rl_help_obj.engine_obj = self.engine_obj
        self.rl_help_obj.route_obj = self.route_obj

    def reset(self, seed: int = None, options: dict = None):  # pylint: disable=arguments-differ
        super().reset(seed=seed)
        self.rl_props['arrival_list'] = list()
        self.rl_props['depart_list'] = list()

        # TODO: Does this make sense?
        if self.optimize or self.optimize is None:
            self.iteration = 0
            self.setup()

            if not self.sim_dict['is_training']:
                self._load_models()

        self._init_props_envs()
        if seed is None:
            # seed = self.iteration
            seed = 0

        self.rl_help_obj.reset_reqs_dict(seed=seed)
        obs = self._get_obs()
        info = self._get_info()
        return obs, info


def _run_iters(env: object, sim_dict: dict, is_training: bool):
    completed_episodes = 0
    while True:
        if is_training:
            obs, curr_reward, is_terminated, is_truncated, curr_info = env.step([0])
        else:
            raise NotImplementedError
        if completed_episodes >= sim_dict['max_iters']:
            break
        if is_terminated or is_truncated:
            _, _ = env.reset()
            completed_episodes += 1
            print(f'{completed_episodes} episodes completed out of {sim_dict["max_iters"]}.')


def _run_testing(env: object, sim_dict: dict):
    env.path_agent.load_model(model_path=sim_dict['path_model'])
    env.core_agent.load_model(model_path=sim_dict['core_model'])
    env.spectrum_agent.load_model(model_path=sim_dict['spectrum_model'])


def _get_model(algorithm: str, device: str, env: object):
    model = None
    yaml_dict = None
    env_name = None
    if algorithm == 'dqn':
        model = None
    elif algorithm == 'ppo':
        yaml_path = os.path.join('sb3_scripts', 'yml', 'ppo.yml')
        yaml_dict = parse_yaml_file(yaml_path)
        env_name = list(yaml_dict.keys())[0]
        kwargs_dict = eval(yaml_dict[env_name]['policy_kwargs'])
        model = PPO(env=env, device=device, policy=yaml_dict[env_name]['policy'],
                    n_steps=yaml_dict[env_name]['n_steps'],
                    batch_size=yaml_dict[env_name]['batch_size'], gae_lambda=yaml_dict[env_name]['gae_lambda'],
                    gamma=yaml_dict[env_name]['gamma'], n_epochs=yaml_dict[env_name]['n_epochs'],
                    vf_coef=yaml_dict[env_name]['vf_coef'], ent_coef=yaml_dict[env_name]['ent_coef'],
                    max_grad_norm=yaml_dict[env_name]['max_grad_norm'],
                    learning_rate=yaml_dict[env_name]['learning_rate'], clip_range=yaml_dict[env_name]['clip_range'],
                    policy_kwargs=kwargs_dict)
    elif algorithm == 'a2c':
        model = None

    return model, yaml_dict[env_name]


def _print_info(sim_dict: dict):
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


def _run(env: object, sim_dict: dict):
    _print_info(sim_dict=sim_dict)

    if sim_dict['is_training']:
        if sim_dict['path_algorithm'] == 'q_learning' or sim_dict['core_algorithm'] == 'q_learning':
            _run_iters(env=env, sim_dict=sim_dict, is_training=True)
        elif sim_dict['spectrum_algorithm'] in ('dqn', 'ppo', 'a2c'):
            model, yaml_dict = _get_model(algorithm=sim_dict['spectrum_algorithm'], device=sim_dict['device'], env=env)
            model.learn(total_timesteps=yaml_dict['n_timesteps'], log_interval=sim_dict['print_step'],
                        callback=sim_dict['callback'])

            save_fp = os.path.join('logs', 'ppo', env.modified_props['network'], env.modified_props['date'],
                                   env.modified_props['sim_start'], 'ppo_model.zip')
            model.save(save_fp)
        else:
            raise ValueError(f'Invalid algorithm received or all algorithms are not reinforcement learning. '
                             f'Expected: q_learning, dqn, ppo, a2c, Got: {sim_dict["path_algorithm"]}, '
                             f'{sim_dict["core_algorithm"]}, {sim_dict["spectrum_algorithm"]}')
    else:
        # TODO: Need to load the model here? Model predict should call everything and will revolve around everything
        #  then
        _run_iters(env=env, sim_dict=sim_dict, is_training=False)


def _setup_rl_sim():
    args_obj = parse_args()
    config_path = os.path.join('ini', 'run_ini', 'config.ini')
    sim_dict = read_config(args_obj=args_obj, config_path=config_path)

    return sim_dict


def run_rl_sim():
    callback = GetModelParams()
    env = SimEnv(render_mode=None, custom_callback=callback, sim_dict=_setup_rl_sim())
    env.sim_dict['callback'] = callback
    _run(env=env, sim_dict=env.sim_dict)


if __name__ == '__main__':
    run_rl_sim()
