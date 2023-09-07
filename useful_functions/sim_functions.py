# Standard library imports
from typing import List

# Third-party library imports
import networkx as nx

# Local application imports
from sim_scripts.routing import Routing


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


def find_path_len(path: List[str], topology: nx.Graph):
    """
    Finds the length of a path in a physical topology.

    :param path: A list of integers representing the nodes in the path.
    :type path: list of str

    :param topology: A networkx graph object representing the physical topology of the simulation.
    :type topology: networkx.Graph

    :return: The length of the path.
    """
    path_len = 0
    for i in range(len(path) - 1):
        path_len += topology[path[i]][path[i + 1]]['length']

    return path_len


def get_route(source, destination, topology, net_spec_db, mod_per_bw, chosen_bw, guard_slots, beta, route_method, ai_obj):
    routing_obj = Routing(source=source, destination=destination,
                          topology=topology, net_spec_db=net_spec_db,
                          mod_formats=mod_per_bw[chosen_bw], bandwidth=chosen_bw,
                          guard_slots=guard_slots)

    # TODO: Change constant QPSK modulation formats
    if route_method == 'nli_aware':
        slots_needed = mod_per_bw[chosen_bw]['QPSK']['slots_needed']

        routing_obj.slots_needed = slots_needed
        routing_obj.beta = beta
        selected_path, path_mod = routing_obj.nli_aware()
    elif route_method == 'xt_aware':
        # TODO: Add xt_type to the configuration file
        selected_path, path_mod = routing_obj.xt_aware(beta=beta, xt_type='with_length')
    elif route_method == 'least_congested':
        selected_path = routing_obj.least_congested_path()
        # TODO: Constant QPSK for now
        path_mod = 'QPSK'
    elif route_method == 'shortest_path':
        selected_path, path_mod = routing_obj.shortest_path()
    elif route_method == 'ai':
        # Used for routing related to artificial intelligence
        selected_path = ai_obj.route(source=int(source), destination=int(destination),
                                     net_spec_db=net_spec_db, chosen_bw=chosen_bw,
                                     guard_slots=guard_slots)

        # A path could not be found, assign None to path modulation
        if not selected_path:
            path_mod = None
        else:
            path_len = find_path_len(path=selected_path, topology=topology)
            path_mod = get_path_mod(mod_formats=mod_per_bw[chosen_bw], path_len=path_len)
    else:
        raise NotImplementedError(f'Routing method not recognized, got: {route_method}.')

    return selected_path, path_mod
