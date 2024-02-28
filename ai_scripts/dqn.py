import os
from datetime import datetime

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from data_scripts.generate_data import create_bw_info
from config_scripts.parse_args import parse_args
from config_scripts.setup_config import read_config
from sim_scripts.engine import Engine


# TODO: Goal is to have this script function like run_sim.py
# TODO: After implementation, add this structure to standards
#   and guidelines document.
# TODO: This will take over SDN, make sure to move helper functions
#   - Most likely needs it's own net spec db? or use engines

class DQNEnv(gym.Env):
    metadata = None

    def __init__(self, render_mode: str = None):
        super().__init__()

        self.engine_obj = None
        self.setup()
        # TODO: Adopt this in other scripts...?
        self.engine_props = self.engine_obj.engine_props

        # TODO: Check the shape
        # TODO: High set to 10?
        self.observation_space = spaces.Box(low=0.0, high=10.0,
                                            shape=(self.engine_props['cores_per_link'],
                                                   self.engine_props['spectral_slots']),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(self.engine_props['spectral_slots'])

        # TODO: Handle this carefully
        self.iteration = 1

    # TODO: Need to save the model? I think RL Zoo will do that for me
    @staticmethod
    def _check_terminated():
        terminated = None
        return terminated

    @staticmethod
    def _calculate_reward(was_routed: bool):
        return 1 if was_routed else -1

    def step(self, action: int):
        was_routed, new_state = None
        success, new_state = self.simulator.allocate_request(action)
        reward = self._calculate_reward(success)

        terminated = self._check_terminated(new_state)
        truncated = False
        info = {}

        return new_state, reward, terminated, truncated, info

    # TODO: Need initial observation, we don't have to use all of the observation!
    # TODO: Don't forget about end iter!
    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed)

        self.engine_obj.init_iter(iteration=self.iteration)
        self.engine_obj.create_topology()
        self.engine_obj.generate_requests()
        # TODO: Input parameters
        self.engine_obj.handle_request()

        obs, info = None
        return obs, info

    # TODO: Make brake up into multiple methods
    def setup(self):
        args_obj = parse_args()
        config_path = os.path.join('..', 'ini', 'run_ini', 'config.ini')
        dqn_sim_dict = read_config(args_obj=args_obj, config_path=config_path)

        sim_start = datetime.now().strftime("%m%d_%H_%M_%S_%f")
        dqn_sim_dict['s1']['date'] = sim_start.split('_')[0]
        tmp_list = sim_start.split('_')
        time_string = f'{tmp_list[1]}_{tmp_list[2]}_{tmp_list[3]}_{tmp_list[4]}'
        dqn_sim_dict['s1']['sim_start'] = time_string
        # TODO: Threading not supported or advised by SB3
        dqn_sim_dict['s1']['thread_num'] = 's1'

        self.engine_obj = Engine(engine_props=dqn_sim_dict['s1'])
        self.engine_obj.engine_props['mod_per_bw'] = create_bw_info(sim_type=dqn_sim_dict['s1']['sim_type'])


# TODO: Probably will have a configuration method
# TODO: Need to move functions from run sim to save and stuff
if __name__ == '__main__':
    dqn_env = DQNEnv()

    obs, info = dqn_env.reset()
