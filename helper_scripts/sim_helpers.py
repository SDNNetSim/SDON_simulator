# Standard library imports
import copy
from typing import List

# Third-party library imports
import networkx as nx
import numpy as np


def get_path_mod(mods_dict: dict, path_len: int):
    """
    Given an object of modulation formats and maximum lengths, choose the one that satisfies the requirements.

    :param mod_formats: The modulation object, holds needed information for maximum reach
    :type mod_formats: dict

    :param path_len: The length of the path to be taken
    :type path_len: int

    :return: The chosen modulation format, or false
    """
    if mods_dict['QPSK']['max_length'] >= path_len > mods_dict['16-QAM']['max_length']:
        resp = 'QPSK'
    elif mods_dict['16-QAM']['max_length'] >= path_len > mods_dict['64-QAM']['max_length']:
        resp = '16-QAM'
    elif mods_dict['64-QAM']['max_length'] >= path_len:
        resp = '64-QAM'
    else:
        return False

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
    sorted_items = sorted(original_dict.items(), key=lambda x: x[1][nested_key])
    sorted_dict = dict(sorted_items)
    return sorted_dict


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


def find_path_len(path_list: List[str], topology: nx.Graph):
    """
    Finds the length of a path in a physical topology.

    :param path: A list of integers representing the nodes in the path.
    :type path: list of str

    :param topology: A networkx graph object representing the physical topology of the simulation.
    :type topology: networkx.Graph

    :return: The length of the path.
    """
    path_len = 0
    for i in range(len(path_list) - 1):
        path_len += topology[path_list[i]][path_list[i + 1]]['length']

    return path_len


def find_path_congestion(path: List[str], network_db):
    """
    Finds the average percentage of congestion for a given path.

    :param path: The path to be analyzed.
    :type path: list

    :param network_db: The current up-to-date network spectrum databse.
    :type network_db: dict

    :return: The average congestion as a decimal.
    :rtype: float
    """
    # Divide by the total length of that array
    cong_per_link = list()
    for src, dest in zip(path, path[1:]):
        src_dest = (src, dest)
        cores_matrix = network_db[src_dest]['cores_matrix']
        cores_per_link = float(len(cores_matrix))

        # Every core will have the same number of spectral slots
        total_slots = len(cores_matrix[0])
        slots_taken = 0

        for curr_core in cores_matrix:
            core_slots_taken = float(len(np.where(curr_core != 0.0)[0]))
            slots_taken += core_slots_taken

        cong_per_link.append(slots_taken / (total_slots * cores_per_link))

    average_path_cong = np.mean(cong_per_link)
    return average_path_cong


def find_core_frag_cong(net_spec_db, path: list, core: int):
    frag_resp = 0.0
    cong_resp = 0.0
    for src, dest in zip(path, path[1:]):
        src_dest = (src, dest)
        core_arr = net_spec_db[src_dest]['cores_matrix'][core]

        if len(core_arr) < 256:
            raise NotImplementedError('Only works for 256 spectral slots.')

        cong_resp += len(np.where(core_arr != 0)[0])

        count = 0
        in_zero_group = False

        for number in core_arr:
            if number == 0:
                if not in_zero_group:
                    in_zero_group = True
            else:
                if in_zero_group:
                    count += 1
                    in_zero_group = False

        frag_resp += count

    num_links = len(path) - 1
    # The lowest number of slots a request can take is 2, the max number of times [1, 1, 0, 2, 2, 0, ..., 5, 5, 0]
    # fragmentation can happen is 85
    frag_resp = (frag_resp / 85.0 / num_links)
    cong_resp = (cong_resp / 256.0 / num_links)
    return frag_resp, cong_resp


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


def find_free_channels(net_spec_dict: dict, slots_needed: int, link_tuple: tuple):
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


def find_taken_channels(net_spec_dict: dict, link_tuple: tuple):
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
