# Standard library imports
import copy
from collections import OrderedDict
from typing import List

# Third-party library imports
import networkx as nx
import numpy as np

# Local application imports
import sim_scripts.spectrum_assignment
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

def filter_mod(mod_formats: dict, path_len: int):
    """
    Given an object of modulation formats and maximum lengths, choose the one that satisfies the requirements.

    :param mod_formats: The modulation object, holds needed information for maximum reach
    :type mod_formats: dict

    :param path_len: The length of the path to be taken
    :type path_len: int

    :return: The filtered mod formats based on length
    """
    mod_formats = sort_nested_dict_vals(mod_formats, nested_key='max_length')
    resp = {key: value for key, value in mod_formats.items() if value['max_length'] >= path_len}
    resp = [key for key, value in mod_formats.items() if value['max_length'] >= path_len]
    return resp
    
def find_max_length(source: int, destination: int, topology: nx.Graph):
    """
    Find the maximum path length possible.

    :param source: The source node.
    :type source: int

    :param destination: The destination node.
    :type destination: int

    :param topology: The network topology.
    :type topology: nx.Graph

    :return: The length of the longest path possible.
    :rtype: float
    """
    paths = list(nx.shortest_simple_paths(topology, source, destination))
    longest_path = paths[-1]
    resp = find_path_len(path=longest_path, topology=topology)

    return resp


def sort_nested_dict_vals(original_dict: dict, nested_key: str):
    """
    Sort a dictionary by a value which belongs to a nested key.

    :param original_dict: The original dictionary.
    :type original_dict: dict

    :param nested_key: The nested key to sort by.
    :type nested_key: str

    :return: The sorted dictionary (ascending).
    :rtype: dict
    """
    sorted_dict = OrderedDict(sorted(original_dict.items(), key=lambda x: x[1][nested_key]))

    return dict(sorted_dict)


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


def get_channel_overlaps(free_channels: dict, free_slots: dict):
    """
    Given the free channels and free slots on a given path, find the number of overlapping and non-overlapping channels
    between adjacent cores.

    :param free_channels: The free channels found on the given path.
    :type free_channels: dict

    :param free_slots: All free slots on the path.
    :type free_slots: dict

    :return: The overlapping and non-overlapping channels for every core.
    :rtype: dict
    """
    resp = {'overlap_channels': {}, 'other_channels': {}}
    num_cores = int(len(free_channels.keys()))

    for core_num, channels in free_channels.items():
        resp['overlap_channels'][core_num] = list()
        resp['other_channels'][core_num] = list()

        for curr_channel in channels:
            overlap = False
            for sub_core in range(0, num_cores):
                if sub_core == core_num:
                    continue

                for _, slots_dict in free_slots.items():
                    # The final core overlaps with all other cores
                    if core_num == num_cores - 1:
                        result = np.isin(curr_channel, slots_dict[sub_core])
                    else:
                        # Only certain cores neighbor each other on a fiber
                        first_neighbor = 5 if core_num == 0 else core_num - 1
                        second_neighbor = 0 if core_num == 5 else core_num + 1

                        result = np.isin(curr_channel, slots_dict[first_neighbor])
                        result = np.append(result, np.isin(curr_channel, slots_dict[second_neighbor]))
                        result = np.append(result, np.isin(curr_channel, slots_dict[num_cores - 1]))

                    if np.any(result):
                        resp['overlap_channels'][core_num].append(curr_channel)
                        overlap = True
                        break

                    resp['other_channels'][core_num].append(curr_channel)

                # No need to check other cores, we already determined this channel overlaps with other channels
                if overlap:
                    break

    return resp


def find_free_slots(net_spec_db: dict, des_link: tuple):
    """
    Find every unallocated spectral slot for a given link.

    :param net_spec_db: The most updated network spectrum database.
    :type net_spec_db: dict

    :param des_link: The link to find the free slots on.
    :type des_link: tuple

    :return: The indexes of the free spectral slots on the link for each core.
    :rtype: dict
    """
    link = net_spec_db[des_link]['cores_matrix']
    resp = {}
    for core_num in range(len(link)):  # pylint: disable=consider-using-enumerate
        indexes = np.where(link[core_num] == 0)[0]
        resp.update({core_num: indexes})

    return resp


def find_free_channels(net_spec_db: dict, slots_needed: int, des_link: tuple):
    """
    Finds the free super-channels on a given link.

    :param net_spec_db: The most updated network spectrum database.
    :type net_spec_db: dict

    :param slots_needed: The number of slots needed for the request.
    :type slots_needed: int

    :param des_link: The link to search on.
    :type des_link: tuple

    :return: A matrix containing the indexes for available super-channels for that request for every core.
    :rtype: dict
    """
    resp = {}
    cores_matrix = copy.deepcopy(net_spec_db[des_link]['cores_matrix'])
    for core_num, link in enumerate(cores_matrix):
        indexes = np.where(link == 0)[0]
        channels = []
        curr_channel = []

        for i, free_index in enumerate(indexes):
            if i == 0:
                curr_channel.append(free_index)
            elif free_index == indexes[i - 1] + 1:
                curr_channel.append(free_index)
                if len(curr_channel) == slots_needed:
                    channels.append(curr_channel.copy())
                    curr_channel.pop(0)
            else:
                curr_channel = [free_index]

        resp.update({core_num: channels})

    return resp


def find_taken_channels(net_spec_db: dict, des_link: tuple):
    """
    Finds the taken super-channels on a given link.

    :param net_spec_db: The most updated network spectrum database.
    :type net_spec_db: dict

    :param des_link: The link to search on.
    :type des_link: tuple

    :return: A matrix containing the indexes for unavailable super-channels for that request for every core.
    :rtype: dict
    """
    resp = {}
    cores_matrix = copy.deepcopy(net_spec_db[des_link]['cores_matrix'])
    for core_num, link in enumerate(cores_matrix):
        channels = []
        curr_channel = []

        for value in link:
            if value > 0:
                curr_channel.append(value)
            elif value < 0 and curr_channel:
                channels.append(curr_channel)
                curr_channel = []

        if curr_channel:
            channels.append(curr_channel)

        resp[core_num] = channels

    return resp


def get_route(properties: dict, source: str, destination: str, topology: nx.Graph, net_spec_db: dict, chosen_bw: str,
              ai_obj: object):
    """
    Given request information, attempt to find a route for the request for various routing methods.

    :param properties: Contains various simulation configuration properties that are constant.
    :type properties: dict

    :param source: The source node.
    :type source: str

    :param destination: The destination node.
    :type destination: str

    :param topology: The network topology information.
    :type topology: nx.Graph

    :param net_spec_db: The network spectrum database.
    :type net_spec_db: dict

    :param chosen_bw: The chosen bandwidth.
    :type chosen_bw: str

    :param ai_obj: The object for artificial intelligence, if it's being used.
    :type ai_obj: object

    :return: The potential paths and path modulation formats.
    :rtype: dict
    """
    # The artificial intelligence objects have their own routing class
    if properties['ai_algorithm'] == 'None':
        routing_obj = Routing(print_warn=properties['warnings'], source=source, destination=destination,
                              topology=topology, net_spec_db=net_spec_db,
                              mod_formats=properties['mod_per_bw'][chosen_bw], bandwidth=chosen_bw,
                              guard_slots=properties['guard_slots'])

    # TODO: Change constant QPSK modulation formats
    if properties['route_method'] == 'nli_aware':
        slots_needed = properties['mod_per_bw'][chosen_bw]['QPSK']['slots_needed']
        routing_obj.slots_needed = slots_needed
        routing_obj.beta = properties['beta']
        resp = routing_obj.nli_aware()
    elif properties['route_method'] == 'xt_aware':
        # TODO: Add xt_type to the configuration file
        selected_path = routing_obj.xt_aware(beta=properties['beta'], xt_type=properties['xt_type'])
        path_len = find_path_len(path= selected_path[0][0], topology=topology)
        # mod = filter_mod(mod_formats=properties['mod_per_bw'][chosen_bw], path_len=path_len)
        temp_mod = sort_nested_dict_vals(properties['mod_per_bw'][chosen_bw], nested_key='max_length')
        resp = [selected_path[0][0]], [list(temp_mod.keys())], selected_path[2]
    elif properties['route_method'] == 'least_congested':
        resp = routing_obj.least_congested_path()
    elif properties['route_method'] == 'shortest_path':
        resp = routing_obj.least_weight_path(weight='length')
    elif properties['route_method'] == 'k_shortest_path':
        resp = routing_obj.k_shortest_path(k_paths=properties['k_paths'])
        temp_mod = sort_nested_dict_vals(properties['mod_per_bw'][chosen_bw], nested_key='max_length')
        paths_mod = list(temp_mod.keys())
        resp = resp[0], [paths_mod] * len(resp[0]), resp[2]
    elif properties['route_method'] == 'ai':
        # Used for routing related to artificial intelligence
        selected_path = ai_obj.route(source=int(source), destination=int(destination),
                                     net_spec_db=net_spec_db, chosen_bw=chosen_bw,
                                     guard_slots=properties['guard_slots'])

        # A path could not be found, assign None to path modulation
        if not selected_path:
            resp = [selected_path], [False], [False]
        else:
            path_len = find_path_len(path=selected_path, topology=topology)
            path_mod = [get_path_mod(mod_formats=properties['mod_per_bw'][chosen_bw], path_len=path_len)]
            resp = [selected_path], [path_mod], [path_len]
    else:
        raise NotImplementedError(f"Routing method not recognized, got: {properties['route_method']}.")

    return resp


def get_spectrum(properties: dict, chosen_bw: str, path: list, net_spec_db: dict, modulation: str, snr_obj: object,
                 path_mod: str):
    """
    Given relevant request information, find a given spectrum for various allocation methods.

    :param properties: Contains various simulation configuration properties that are constant.
    :type properties: dict

    :param chosen_bw: The chosen bandwidth for this request.
    :type chosen_bw: str

    :param path: The chosen path for this request.
    :type path: list

    :param net_spec_db: The network spectrum database.
    :type net_spec_db: dict

    :param modulation: The modulation format chosen for this request.
    :type modulation: str

    :param snr_obj: If check_snr is true, the object containing all snr related methods.
    :type snr_obj: object

    :param path_mod: The modulation format for the given path.
    :type path_mod: str

    :return: The information related to the spectrum found for allocation, false otherwise.
    :rtype: dict
    """
    slots_needed = properties['mod_per_bw'][chosen_bw][modulation]['slots_needed']
    spectrum_assignment = sim_scripts.spectrum_assignment.SpectrumAssignment(print_warn=properties['warnings'],
                                                                             path=path, slots_needed=slots_needed,
                                                                             net_spec_db=net_spec_db,
                                                                             guard_slots=properties['guard_slots'],
                                                                             is_sliced=False,
                                                                             alloc_method=properties[
                                                                                 'allocation_method'])

    spectrum = spectrum_assignment.find_free_spectrum()
    xt_cost = None

    if spectrum is not False:
        if properties['check_snr'] != 'None':
            _update_snr_obj(snr_obj=snr_obj, spectrum=spectrum, path=path, path_mod=path_mod,
                            spectral_slots=properties['spectral_slots'], net_spec_db=net_spec_db)
            snr_check, xt_cost = handle_snr(check_snr=properties['check_snr'], snr_obj=snr_obj)

            if not snr_check:
                return False, 'xt_threshold', xt_cost

        # No reason for blocking, return spectrum and None
        return spectrum, None, xt_cost

    return False, 'congestion', xt_cost


def _update_snr_obj(snr_obj: object, spectrum: dict, path: list, path_mod: str, spectral_slots: int, net_spec_db: dict):
    """
    Updates variables in the signal-to-noise ratio calculation object.

    :param snr_obj: The object whose variables are updated.
    :type snr_obj: object

    :param spectrum: The spectrum chosen for the request.
    :type spectrum: dict

    :param path: The chosen path for the request.
    :type path: list

    :param path_mod: The modulation format chosen for the request.
    :type path_mod: str

    :param spectral_slots: The total number of spectral slots for each core in the network.
    :type spectral_slots: int

    :param net_spec_db: The network spectrum database.
    :type net_spec_db: dict
    """
    snr_obj.path = path
    snr_obj.path_mod = path_mod
    snr_obj.spectrum = spectrum
    snr_obj.assigned_slots = spectrum['end_slot'] - spectrum['start_slot'] + 1
    snr_obj.spectral_slots = spectral_slots
    snr_obj.net_spec_db = net_spec_db


def handle_snr(check_snr: str, snr_obj: object):
    """
    Determines which type of signal-to-noise ratio calculation is used and calculates it.

    :param check_snr: The type of SNR calculation for every request.
    :type check_snr: str

    :param snr_obj: Object containing all methods for SNR calculation.
    :type snr_obj: object

    :return: If the SNR threshold can be met or not.
    :rtype: bool
    """
    if check_snr == "snr_calculation_nli":
        snr_check, xt_cost = snr_obj.check_snr()
    elif check_snr == "xt_calculation":
        snr_check, xt_cost = snr_obj.check_xt()
    elif check_snr == "snr_calculation_xt":
        snr_check, xt_cost = snr_obj.check_snr_xt()
    else:
        raise NotImplementedError(f'Unexpected check_snr flag got: {check_snr}')

    return snr_check, xt_cost
