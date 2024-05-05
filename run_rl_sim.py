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
# TODO: ONLY use props in each object, do not duplicate here (Write in a standard?)
class SimEnv(gym.Env):  # pylint: disable=abstract-method
    """
    Simulates a deep q-learning environment with stable baselines3 integration.
    """
    metadata = dict()

    # TODO: Double check constructors used, probably won't need many now
    # TODO: Double check to see if props needs to be reset...
    def __init__(self, render_mode: str = None, custom_callback: object = None, sim_dict: dict = None, **kwargs):
        super().__init__()

        self.rl_props = copy.deepcopy(empty_rl_props)
        self.rl_props['super_channel_space'] = None
        self.q_props = copy.deepcopy(empty_q_props)
        self.drl_props = copy.deepcopy(empty_drl_props)

        self.sim_dict = sim_dict['s1']

        self.iteration = 0
        self.options = None
        self.optimize = None
        self.callback = custom_callback
        self.render_mode = render_mode

        self.engine_obj = None
        self.route_obj = None
        # TODO: These are no longer the updated props...(Q props, drl props)
        self.rl_help_obj = RLHelpers(rl_props=self.rl_props, engine_obj=self.engine_obj, route_obj=self.route_obj,
                                     q_props=self.q_props, drl_props=self.drl_props)

        # TODO: Core and spectrum agents
        # TODO: I have self.engine_props and then engine props in the actual object...
        self.path_agent = PathAgent(path_algorithm=self.sim_dict['path_algorithm'], rl_props=self.rl_props,
                                    rl_help_obj=self.rl_help_obj)
        self.core_agent = CoreAgent(core_algorithm=self.sim_dict['core_algorithm'], rl_props=self.rl_props,
                                    rl_help_obj=self.rl_help_obj)
        # TODO: Hard coded algorithm, change
        self.spectrum_agent = SpectrumAgent(spectrum_algorithm='ppo', rl_props=self.rl_props)

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

        if self.sim_dict['path_algorithm'] == 'q_learning' and self.sim_dict['is_training']:
            self.rl_help_obj.q_props = self.q_props
        elif self.sim_dict['core_algorithm'] == 'q_learning' and self.sim_dict['is_training']:
            pass
        else:
            self.rl_help_obj.drl_props = self.drl_props

        self.rl_help_obj.ai_props = self.rl_props
        self.rl_help_obj.engine_obj = self.engine_obj
        self.rl_help_obj.handle_releases()
        self.rl_help_obj.update_route_props(chosen_path=self.rl_props['chosen_path'], bandwidth=bandwidth)

    def step(self, action: list):
        req_info_dict = self.rl_props['arrival_list'][self.rl_props['arrival_count']]
        req_id = req_info_dict['req_id']
        bandwidth = req_info_dict['bandwidth']
        self._update_helper_obj(action=action, bandwidth=bandwidth)
        self.rl_help_obj.allocate(route_obj=self.route_obj)
        reqs_status_dict = self.engine_obj.reqs_status_dict

        if req_id in reqs_status_dict:
            was_allocated = True
        else:
            was_allocated = False
        self.rl_help_obj.update_snapshots()

        # TODO: Change
        # drl_reward = self.rl_help_obj.calculate_drl_reward(was_allocated=was_allocated)
        drl_reward = 1

        if self.sim_dict['path_algorithm'] == 'q_learning' and self.sim_dict['is_training']:
            self.path_agent.update(was_allocated=was_allocated, net_spec_dict=self.engine_obj.net_spec_dict,
                                   iteration=self.iteration)
        elif self.sim_dict['core_algorithm'] == 'q_learning' and self.sim_dict['is_training']:
            self.core_agent.update(was_allocated=was_allocated, net_spec_dict=self.engine_obj.net_spec_dict,
                                   iteration=self.iteration)
        self.rl_props['arrival_count'] += 1
        terminated = self._check_terminated()
        new_obs = self._get_obs()
        truncated = False
        info = self._get_info()

        return new_obs, drl_reward, terminated, truncated, info

    @staticmethod
    def _get_info():
        return dict()

    # TODO: Split this up into more functions
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
        # TODO: Will probably have to change for test/train
        if self.sim_dict['path_algorithm'] == 'q_learning' and self.sim_dict['is_training']:
            self.path_agent.get_route()
            self.rl_help_obj.rl_props['chosen_path'] = [self.rl_props['chosen_path']]
            self.route_obj.route_props['paths_list'] = self.rl_help_obj.rl_props['chosen_path']
            path_len = find_path_len(path_list=self.rl_props['paths_list'][self.rl_props['path_index']],
                                     topology=self.engine_obj.topology)
            path_mod = get_path_mod(mods_dict=curr_req['mod_formats'], path_len=path_len)
            self.rl_props['core_index'] = None
        elif self.sim_dict['core_algorithm'] == 'q_learning' and self.sim_dict['is_training']:
            self.route_obj.sdn_props = self.rl_props['mock_sdn_dict']
            self.route_obj.engine_props['route_method'] = 'shortest_path'
            self.route_obj.get_route()
            self.rl_props['chosen_path'] = self.route_obj.route_props['paths_list']
            # TODO: Always shortest path
            self.rl_props['path_index'] = 0
            path_mod = self.route_obj.route_props['mod_formats_list'][0][0]
            self.core_agent.get_core()
        # TODO: Modify
        else:
            self.route_obj.sdn_props = self.rl_props['mock_sdn_dict']
            self.route_obj.get_route()
            self.rl_props['paths_list'] = self.route_obj.route_props['paths_list']
            self.rl_props['path_index'] = 0
            self.rl_props['core_index'] = None
            path_mod = self.route_obj.route_props['mod_formats_list'][0][0]

        if path_mod is not False:
            slots_needed = curr_req['mod_formats'][path_mod]['slots_needed']
        else:
            slots_needed = 0
        # super_channels = self.rl_help_obj.get_super_channels(slots_needed=slots_needed,
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

    def _init_envs(self):
        if self.sim_dict['path_algorithm'] == 'q_learning' and self.sim_dict['is_training']:
            self.path_agent.engine_props = self.engine_obj.engine_props
            self.path_agent.setup_env()
        elif self.sim_dict['core_algorithm'] == 'q_learning' and self.sim_dict['is_training']:
            self.core_agent.engine_props = self.engine_obj.engine_props
            self.core_agent.setup_env()
        # TODO: Init everything? What to actually do when testing?
        else:
            raise NotImplementedError

    def _create_input(self):
        base_fp = os.path.join('data')
        self.sim_dict['thread_num'] = 's1'
        # Added only for structure consistency
        get_start_time(sim_dict={'s1': self.sim_dict})
        file_name = "sim_input_s1.json"

        self.engine_obj = Engine(engine_props=self.sim_dict)
        self.route_obj = Routing(engine_props=self.engine_obj.engine_props,
                                 sdn_props=self.rl_props['mock_sdn_dict'])

        # TODO: Keep an eye on this, used to be sim dict but now I'm confused as to what it actually does
        self.sim_props = create_input(base_fp=base_fp, engine_props=self.sim_dict)
        modified_props = copy.deepcopy(self.sim_props)
        if 'topology' in self.sim_props:
            modified_props.pop('topology')
            modified_props.pop('callback')

        save_input(base_fp=base_fp, properties=modified_props, file_name=file_name,
                   data_dict=modified_props)

    def setup(self):
        """
        Sets up this class.
        """
        # TODO: Not sure if I still need these props here (rl_props)...
        self.optimize = self.sim_dict['optimize']
        self.rl_props['k_paths'] = self.sim_dict['k_paths']
        self.rl_props['cores_per_link'] = self.sim_dict['cores_per_link']
        self.rl_props['spectral_slots'] = self.sim_dict['spectral_slots']

        self._create_input()

        start_arr_rate = float(self.sim_dict['arrival_rate']['start'])
        self.engine_obj.engine_props['erlang'] = start_arr_rate / self.sim_dict['holding_time']
        self.engine_obj.engine_props['arrival_rate'] = start_arr_rate * self.sim_dict['cores_per_link']

    # TODO: Split this up into more functions
    def reset(self, seed: int = None, options: dict = None):  # pylint: disable=arguments-differ
        super().reset(seed=seed)
        self.rl_props['arrival_list'] = list()
        self.rl_props['depart_list'] = list()

        # TODO: Doesn't really make sense in the config file
        if self.optimize or self.optimize is None:
            # TODO: These will have to be modified
            # TODO: Not sure if q and drl props are needed?
            self.rl_props['q_props'] = copy.deepcopy(empty_q_props)
            self.rl_props['drl_props'] = copy.deepcopy(empty_drl_props)
            self.iteration = 0
            self.setup()

        self.rl_props['arrival_count'] = 0
        self.engine_obj.init_iter(iteration=self.iteration)
        self.engine_obj.create_topology()
        self.rl_help_obj.topology = self.engine_obj.topology
        self.rl_props['num_nodes'] = len(self.engine_obj.topology.nodes)

        if self.iteration == 0:
            self._init_envs()
        # TODO: A reset of props?
        else:
            print('Did not do anything for the next iteration. To be continued/debugged.')

        self.rl_help_obj.rl_props = self.rl_props
        self.rl_help_obj.engine_obj = self.engine_obj
        self.rl_help_obj.route_obj = self.route_obj

        # TODO: Generalize (Ask about this)
        if seed is None:
            # seed = self.iteration
            seed = 0

        self.rl_help_obj.reset_reqs_dict(seed=seed)
        obs = self._get_obs()
        info = self._get_info()
        return obs, info


# In order to have the same structure as DRL
# TODO: Have the above code return an action! Stepping with a random zero makes no sense at all
def _run_iters(env: object, sim_dict: dict):
    completed_episodes = 0
    while True:
        # TODO: Stepping with zero?
        obs, curr_reward, is_terminated, is_truncated, curr_info = env.step([0])
        if completed_episodes >= sim_dict['max_iters']:
            break
        if is_terminated or is_truncated:
            _, _ = env.reset()
            completed_episodes += 1
            print(f'{completed_episodes} episodes completed out of {sim_dict["max_iters"]}.')


def _run_testing(env: object, sim_dict: dict):
    # model = DQN.load('./logs/DQN/best_model.zip', env=env)
    # curr_action, _states = model.predict(obs)
    raise NotImplementedError


# TODO: Type for env
def _get_model(algorithm: str, device: str, env):
    model = None
    if algorithm == 'dqn':
        model = None
    elif algorithm == 'ppo':
        # TODO: Number of training steps
        # TODO: Policy to configuration file
        model = PPO(policy='MlpPolicy', env=env, device=device)
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
        _run_iters(env=env, sim_dict=sim_dict)
    # TODO: Consider RL zoo and optimization via command line
    # TODO: Also register the environment
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


def run_rl_sim():
    callback = GetModelParams()
    env = SimEnv(render_mode=None, custom_callback=callback, sim_dict=_setup_rl_sim())
    env.sim_dict['callback'] = callback

    if env.sim_dict['is_training']:
        _run_training(env=env, sim_dict=env.sim_dict)
    else:
        _run_testing(env=env, sim_dict=env.sim_dict)


if __name__ == '__main__':
    run_rl_sim()
