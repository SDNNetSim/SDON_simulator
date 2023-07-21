# Standard library imports
from typing import List

# Third-party library imports
import networkx as nx

# Local application imports
from ai.reinforcement_learning import *

# TODO: I don't like this at all, objectify these functions
# TODO: Update doc strings
# A class dedicated to artificial intelligence routing and spectrum assignment
AI_OBJ = None


def get_path_mod(mod_formats: dict, path_len: int):
    """
    Given an object of modulation formats and maximum lengths, choose the one that satisfies the requirements.

    :param mod_formats: The modulation object, holds needed information for maximum reach
    :type mod_formats: dict
    :param path_len: The length of the path to be taken
    :type path_len: int

    :return: The chosen modulation format, or false
    """
    if mod_formats['QPSK']['max_length'] >= path_len > mod_formats['16-QAM']['max_length']:
        resp = 'QPSK'
    elif mod_formats['16-QAM']['max_length'] >= path_len > mod_formats['64-QAM']['max_length']:
        resp = '16-QAM'
    elif mod_formats['64-QAM']['max_length'] >= path_len:
        resp = '64-QAM'
    else:
        return False

    return resp


def sort_dict_keys(dictionary: dict):
    """
    Given a dictionary with key-value pairs, return a new dictionary with the same pairs, sorted by keys in descending
    order.

    :param dictionary: The dictionary to sort.
    :type dictionary: dict

    :return: A new dictionary with the same pairs as the input dictionary, but sorted by keys in descending order.
    :rtype: dict
    """
    sorted_keys = sorted(map(int, dictionary.keys()), reverse=True)
    sorted_dict = {str(key): dictionary[str(key)] for key in sorted_keys}

    return sorted_dict


def find_path_len(path: List[int], topology: nx.Graph):
    """
    Finds the length of a path in a physical topology.

    :param path: A list of integers representing the nodes in the path.
    :type path: list of int
    :param topology: A networkx graph object representing the physical topology of the simulation.
    :type topology: networkx.Graph

    :return: The length of the path.
    """
    path_len = 0
    for i in range(len(path) - 1):
        path_len += topology[path[i]][path[i + 1]]['length']

    return path_len


# TODO: Finish
def save_ai_obj(**kwargs):
    global AI_OBJ

    if kwargs['sim_data']['ai_algorithm'] == 'q_learning':
        if kwargs['sim_data']['is_training']:
            pass
        else:
            pass


def update_ai_obj(**kwargs):
    global AI_OBJ

    if kwargs['sim_data']['ai_algorithm'] == 'q_learning':
        if kwargs['update_env']:
            AI_OBJ.update_environment(routed=kwargs['routed'], path=kwargs['path'], free_slots=kwargs['free_slots'])
        else:
            # Decay epsilon for half of the iterations evenly each time
            if 1 <= kwargs['iteration'] <= kwargs['iteration'] // 2 and kwargs['sim_data']['is_training']:
                decay_amount = (AI_OBJ.epsilon / (kwargs['iteration'] // 2) - 1)
                AI_OBJ.decay_epsilon(amount=decay_amount)


def setup_ai_obj(**kwargs):
    global AI_OBJ

    if kwargs['sim_data']['ai_algorithm'] == 'q_learning':
        AI_OBJ = QLearning()
        AI_OBJ.topology = kwargs['topology']
        AI_OBJ.setup_environment()

        if not kwargs['sim_data']['is_training']:
            AI_OBJ.load_table(path=kwargs['sim_info'], max_segments=kwargs['max_segments'])
        else:
            AI_OBJ.load_table(path=kwargs['sim_data']['trained_table'], max_segments=kwargs['max_segments'])

        AI_OBJ.set_seed(seed=kwargs['seed'])
        # Spectrum assignment for AI not supported at this time
        return AI_OBJ.route, None

    else:
        raise NotImplementedError(f"The {kwargs['sim_data']['ai_algorithm']} ai algorithm is not recognized.")
