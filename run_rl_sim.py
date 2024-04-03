import os
import copy

import networkx as nx
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
from arg_scripts.ai_args import empty_drl_props, empty_q_props, empty_ai_props


# TODO: Account for script name change in bash scripts (run_rl_sim)
class SimEnv(gym.Env):  # pylint: disable=abstract-method
    """
    Simulates a deep q-learning environment with stable baselines3 integration.
    """
    metadata = dict()

    def __init__(self, render_mode: str = None, **kwargs):
        super().__init__()
        # TODO: These need to be updated in setup or something
        #   - I don't think I should reset
        self.ai_props = copy.deepcopy(empty_ai_props)
        self.q_props = copy.deepcopy(empty_q_props)
        self.drl_props = copy.deepcopy(empty_drl_props)

        self.sim_dict = dict()
        self.iteration = 0
        self.options = None
        self.optimize = None
        self.render_mode = render_mode

        self.engine_obj = None
        self.route_obj = None
        self.helper_obj = AIHelpers(ai_props=self.ai_props, engine_obj=self.engine_obj, route_obj=self.route_obj,
                                    q_props=self.q_props, drl_props=self.drl_props)

        # Used to get config variables into the observation space
        self.reset(options={'save_sim': False})
        self.helper_obj.find_maximums()

        self.observation_space = self.helper_obj.get_obs_space()
        self.action_space = self.helper_obj.get_action_space()

    # TODO: Routes should still depend on cores, but cores will now depend on spectrum (implement)
    def _get_max_future_q(self):
        q_values = list()
        # TODO: Do NOT forget to update ai props somewhere after each request
        cores_matrix = self.q_props['cores_matrix'][self.ai_props['source']]
        cores_matrix = cores_matrix[self.ai_props['destination']][self.ai_props['path_index']]
        for core_index in range(self.engine_obj.engine_props['cores_per_link']):
            curr_q = cores_matrix[core_index]['q_value']
            q_values.append(curr_q)

        max_index = np.argmax(q_values)
        resp_value = cores_matrix[max_index]['q_value']
        return resp_value

    def _update_routes_matrix(self, was_routed: bool):
        if was_routed:
            reward = 1.0
        else:
            reward = -1.0

        routes_matrix = self.q_props['routes_matrix'][self.ai_props['source']][self.ai_props['destination']]
        current_q = routes_matrix[self.ai_props['path_index']]['q_value']
        max_future_q = self._get_max_future_q()
        delta = reward + self.engine_obj.engine_props['discount_factor'] * max_future_q
        td_error = current_q - (reward + self.engine_obj.engine_props['discount_factor'] * max_future_q)
        # TODO: Remove this call from before, it's called here!
        self.helper_obj.update_q_stats(reward=reward, stats_flag='routes_dict', td_error=td_error,
                                       iteration=self.iteration)

        engine_props = self.engine_obj.engine_props
        new_q = ((1.0 - engine_props['learn_rate']) * current_q) + (engine_props['learn_rate'] * delta)

        # TODO: Make sure this works
        routes_matrix = self.q_props['routes_matrix'][self.ai_props['source']][self.ai_props['destination']]
        routes_matrix[self.ai_props['path_index']]['q_value'] = new_q

    def _update_cores_matrix(self, was_routed: bool):
        if was_routed:
            reward = 1.0
        else:
            reward = -1.0

        q_cores_matrix = self.q_props['cores_matrix'][self.ai_props['source']]
        q_cores_matrix = q_cores_matrix[self.ai_props['destination']][self.ai_props['path_index']]
        current_q = q_cores_matrix[self.ai_props['core_index']]['q_value']
        # TODO: Change
        max_future_q = 1.0
        delta = reward + self.engine_obj.engine_props['discount_factor'] * max_future_q
        self.helper_obj.update_q_stats(reward=reward, stats_flag='cores_dict', td_error=delta, iteration=self.iteration)

        engine_props = self.engine_obj.engine_props
        new_q_core = ((1.0 - engine_props['learn_rate']) * current_q) + (engine_props['learn_rate'] * delta)

        # TODO: Make sure this works
        # TODO: Ensure all these variables are being updates properly
        cores_matrix = self.q_props['cores_matrix'][self.ai_props['source']][self.ai_props['destination']]
        cores_matrix[self.ai_props['path_index']][self.ai_props['core_index']]['q_value'] = new_q_core

    def get_route(self, sdn_props: dict, route_props: dict):
        random_float = np.round(np.random.uniform(0, 1), decimals=1)
        self.ai_props['source'] = int(sdn_props['source'])
        self.ai_props['destination'] = int(sdn_props['destination'])
        routes_matrix = self.q_props['routes_matrix']
        self.ai_props['paths_list'] = routes_matrix[self.ai_props['source']][self.ai_props['source']]['path']

        if self.ai_props['paths_list'].ndim != 1:
            self.ai_props['paths_list'] = self.ai_props['paths_list'][:, 0]

        if random_float < self.q_props['epsilon']:
            self.ai_props['path_index'] = np.random.choice(self.ai_props['k_paths'])

            if self.ai_props['path_index'] == 1 and self.ai_props['k_paths'] == 1:
                self.ai_props['path_index'] = 0
            # TODO: Defined outside constructor
            self.chosen_path = self.ai_props['paths_list'][self.ai_props['path_index']]
        else:
            # TODO: I don't think chosen path is within props
            self.ai_props['path_index'], self.ai_props['chosen_path'] = self.helper_obj.get_max_curr_q()

        if len(self.chosen_path) == 0:
            raise ValueError('The chosen path can not be None')

        # TODO: Not sure about sdn props either...
        self.helper_obj.update_route_props(sdn_props=sdn_props, route_props=route_props, chosen_path=self.chosen_path)

    def get_core(self, spectrum_props: dict):
        random_float = np.round(np.random.uniform(0, 1), decimals=1)

        if random_float < self.q_props['epsilon']:
            self.core_index = np.random.randint(0, self.engine_props['cores_per_link'])
        else:
            q_values = self.q_props['cores_matrix'][self.source][self.destination][self.path_index]['q_value']
            self.core_index = np.argmax(q_values)

        spectrum_props['forced_core'] = self.core_index

    def _init_q_tables(self):
        for source in range(0, self.ai_props['num_nodes']):
            for destination in range(0, self.ai_props['num_nodes']):
                # A node cannot be attached to itself
                if source == destination:
                    continue

                shortest_paths = nx.shortest_simple_paths(G=self.engine_obj.engine_props['topology'],
                                                          source=str(source), target=str(destination), weight='length')
                for k, curr_path in enumerate(shortest_paths):
                    if k >= self.ai_props['k_paths']:
                        break
                    self.q_props['routes_matrix'][source, destination, k] = (curr_path, 0.0)

                    for core_action in range(self.engine_obj.engine_props['cores_per_link']):
                        self.q_props['cores_matrix'][source, destination, k, core_action] = (curr_path, core_action,
                                                                                             0.0)

    def setup_q_env(self):
        self.q_props['epsilon'] = self.engine_obj.engine_props['epsilon_start']
        route_types = [('path', 'O'), ('q_value', 'f8')]
        core_types = [('path', 'O'), ('core_action', 'i8'), ('q_value', 'f8')]

        self.q_props['routes_matrix'] = np.empty((self.ai_props['num_nodes'], self.ai_props['num_nodes'],
                                                  self.ai_props['k_paths']), dtype=route_types)
        self.q_props['cores_matrix'] = np.empty((self.ai_props['num_nodes'], self.ai_props['num_nodes'],
                                                 self.ai_props['k_paths'],
                                                 self.engine_obj.engine_props['cores_per_link']), dtype=core_types)

        self._init_q_tables()

    def _check_terminated(self):
        if self.ai_props['arrival_count'] == (self.engine_obj.engine_props['num_requests']):
            terminated = True
            base_fp = os.path.join('data')
            # TODO: Update amount
            amount = 0
            # self.helper_obj.decay_epsilon(amount=amount, iteration=self.iteration)
            self.engine_obj.end_iter(iteration=self.iteration, print_flag=False, ai_flag=True, base_fp=base_fp)
            self.iteration += 1
        else:
            terminated = False

        return terminated

    def _update_helper_obj(self, action: list):
        # TODO: Make sure these are used in the new q learning functions
        self.helper_obj.path_index = action[0]
        self.helper_obj.core_num = action[1]
        self.helper_obj.slice_request = action[2]

        if self.helper_obj.path_index < 0 or self.helper_obj.path_index > (self.ai_props['k_paths'] - 1):
            raise ValueError(f'Path index out of range: {self.helper_obj.path_index}')
        if self.helper_obj.core_num < 0 or self.helper_obj.core_num > (
                self.ai_props['cores_per_link'] - 1):
            raise ValueError(f'Core index out of range: {self.helper_obj.core_num}')

        # TODO: Double check these updates (ql and drl props)
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

        drl_reward = self.helper_obj.calculate_drl_reward(was_allocated=was_allocated)
        self._update_routes_matrix(was_routed=was_allocated)
        self._update_cores_matrix(was_routed=was_allocated)
        self.ai_props['arrival_count'] += 1
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
        if self.ai_props['arrival_count'] == self.engine_obj.engine_props['num_requests']:
            curr_req = self.ai_props['arrival_list'][self.ai_props['arrival_count'] - 1]
        else:
            curr_req = self.ai_props['arrival_list'][self.ai_props['arrival_count']]

        self.ai_props['mock_sdn_dict'] = self.helper_obj.update_mock_sdn(curr_req=curr_req)
        self.route_obj.sdn_props = self.ai_props['mock_sdn_dict']
        # TODO: Q-Learning to take over getting the route
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
        # TODO: Routing will change to q-learning so make sure to double check that
        self.route_obj = Routing(engine_props=self.engine_obj.engine_props,
                                 sdn_props=self.ai_props['mock_sdn_dict'])
        self.sim_dict['s1'] = create_input(base_fp=base_fp, engine_props=self.sim_dict['s1'])
        save_input(base_fp=base_fp, properties=self.sim_dict['s1'], file_name=file_name,
                   data_dict=self.sim_dict['s1'])

    def setup(self):
        """
        Sets up this class.
        """
        args_obj = parse_args()
        config_path = os.path.join('ini', 'run_ini', 'config.ini')
        self.sim_dict = read_config(args_obj=args_obj, config_path=config_path)

        self.optimize = args_obj['optimize']
        self.ai_props['k_paths'] = self.sim_dict['s1']['k_paths']
        self.ai_props['cores_per_link'] = self.sim_dict['s1']['cores_per_link']
        self.ai_props['spectral_slots'] = self.sim_dict['s1']['spectral_slots']

        self._create_input()
        start_arr_rate = float(self.sim_dict['s1']['arrival_rate']['start'])
        self.engine_obj.engine_props['erlang'] = start_arr_rate / self.sim_dict['s1']['holding_time']
        self.engine_obj.engine_props['arrival_rate'] = start_arr_rate * self.sim_dict['s1']['cores_per_link']

    # TODO: If not in optimize mode, don't create new obj and increment iter
    #   - Make sure to force the seed
    def reset(self, seed: int = None, options: dict = None):  # pylint: disable=arguments-differ
        super().reset(seed=seed)
        if self.optimize or self.optimize is None:
            self.ai_props['q_props'] = copy.deepcopy(empty_q_props)
            self.ai_props['drl_props'] = copy.deepcopy(empty_drl_props)
            self.iteration = 0
            self.setup()

        self.ai_props['arrival_count'] = 0
        self.engine_obj.init_iter(iteration=self.iteration)
        self.engine_obj.create_topology()
        self.ai_props['num_nodes'] = len(self.engine_obj.topology.nodes)

        self.setup_q_env()
        # TODO: QL and DRL props set up correctly?
        self.helper_obj.ai_props = self.ai_props
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


if __name__ == '__main__':
    env = SimEnv(algorithm='PPO')

    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=20000, log_interval=1)

    # model.save('./logs/best_PPO_model.zip')
    # model = DQN.load('./logs/DQN/best_model.zip', env=env)
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
