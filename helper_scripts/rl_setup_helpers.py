import os

from config_scripts.parse_args import parse_args
from config_scripts.setup_config import read_config
from arg_scripts.rl_args import VALID_PATH_ALGORITHMS, VALID_CORE_ALGORITHMS, VALID_SPECTRUM_ALGORITHMS


def setup_rl_sim():
    """
    Set up a reinforcement learning simulation.

    :return: The simulation dictionary for the RL sim.
    :rtype: dict
    """
    args_obj = parse_args()
    config_path = os.path.join('ini', 'run_ini', 'config.ini')
    sim_dict = read_config(args_obj=args_obj, config_path=config_path)

    return sim_dict


def print_info(sim_dict: dict):
    """
    Prints relevant RL simulation information.

    :param sim_dict: Simulation dictionary (engine props).
    """
    if sim_dict['path_algorithm'] in VALID_PATH_ALGORITHMS:
        print(f'Beginning training process for the PATH AGENT using the '
              f'{sim_dict["path_algorithm"].title()} algorithm.')
    elif sim_dict['core_algorithm'] in VALID_CORE_ALGORITHMS:
        print(f'Beginning training process for the CORE AGENT using the '
              f'{sim_dict["core_algorithm"].title()} algorithm.')
    elif sim_dict['spectrum_algorithm'] in VALID_SPECTRUM_ALGORITHMS:
        print(f'Beginning training process for the SPECTRUM AGENT using the '
              f'{sim_dict["spectrum_algorithm"].title()} algorithm.')
    else:
        raise ValueError(f'Invalid algorithm received or all algorithms are not reinforcement learning. '
                         f'Expected: q_learning, dqn, ppo, a2c, Got: {sim_dict["path_algorithm"]}, '
                         f'{sim_dict["core_algorithm"]}, {sim_dict["spectrum_algorithm"]}')
