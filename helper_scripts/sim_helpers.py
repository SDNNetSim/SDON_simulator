import copy

import networkx as nx
import numpy as np


def get_path_mod(mods_dict: dict, path_len: int):
    """
    Choose a modulation format that will allocate a network request.

    :param mods_dict: Information for maximum reach for each modulation format.
    :param path_len: The length of the path to be taken.
    :return: The chosen modulation format.
    :rtype: str
    """
    # Pycharm auto-formats it like this for comparisons...I'd rather this look weird than look at PyCharm warnings
    if mods_dict['QPSK']['max_length'] >= path_len > mods_dict['16-QAM']['max_length']:
        resp = 'QPSK'
    elif mods_dict['16-QAM']['max_length'] >= path_len > mods_dict['64-QAM']['max_length']:
        resp = '16-QAM'
    elif mods_dict['64-QAM']['max_length'] >= path_len:
        resp = '64-QAM'
    else:
        return False

    return resp


def find_max_path_len(source: int, destination: int, topology: nx.Graph):
    """
    Find the maximum path length possible of a path in the network.

    :param source: The source node.
    :param destination: The destination node.
    :param topology: The network topology.
    :return: The length of the longest path possible.
    :rtype: float
    """
    all_paths_list = list(nx.shortest_simple_paths(topology, source, destination))
    path_list = all_paths_list[-1]
    resp = find_path_len(path_list=path_list, topology=topology)

    return resp


def sort_nested_dict_vals(original_dict: dict, nested_key: str):
    """
    Sort a dictionary by a value which belongs to a nested key.

    :param original_dict: The original dictionary.
    :param nested_key: The nested key to sort by.
    :return: The sorted dictionary, ascending.
    :rtype: dict
    """
    sorted_items = sorted(original_dict.items(), key=lambda x: x[1][nested_key])
    sorted_dict = dict(sorted_items)
    return sorted_dict


def sort_dict_keys(dictionary: dict):
    """
    Sort a dictionary by keys in descending order.

    :param dictionary: The dictionary to sort.
    :return: The newly sorted dictionary.
    :rtype: dict
    """
    sorted_keys = sorted(map(int, dictionary.keys()), reverse=True)
    sorted_dict = {str(key): dictionary[str(key)] for key in sorted_keys}

    return sorted_dict


def find_path_len(path_list: list, topology: nx.Graph):
    """
    Finds the length of a path in a physical topology.

    :param path_list: A list of integers representing the nodes in the path.
    :param topology: The network topology.
    :return: The length of the path.
    """
    path_len = 0
    for i in range(len(path_list) - 1):
        path_len += topology[path_list[i]][path_list[i + 1]]['length']

    return path_len


def find_path_cong(path_list: list, net_spec_dict: dict):
    """
    Finds the average percentage of congestion for a given path.

    :param path_list: The path to be analyzed.
    :param net_spec_dict: The current up-to-date network spectrum database.
    :return: The average congestion as a decimal.
    :rtype: float
    """
    # Divide by the total length of that array
    links_cong_list = list()
    for src, dest in zip(path_list, path_list[1:]):
        src_dest = (src, dest)
        cores_matrix = net_spec_dict[src_dest]['cores_matrix']
        cores_per_link = float(len(cores_matrix))

        # Every core will have the same number of spectral slots
        total_slots = len(cores_matrix[0])
        slots_taken = 0
        for curr_core in cores_matrix:
            core_slots_taken = float(len(np.where(curr_core != 0.0)[0]))
            slots_taken += core_slots_taken

        links_cong_list.append(slots_taken / (total_slots * cores_per_link))

    average_path_cong = np.mean(links_cong_list)
    return average_path_cong


def get_channel_overlaps(free_channels_dict: dict, free_slots_dict: dict):
    """
    Find the number of overlapping and non-overlapping channels between adjacent cores.

    :param free_channels_dict: The free super-channels found on a path.
    :param free_slots_dict: The free slots found on the given path.
    :return: The overlapping and non-overlapping channels for every core.
    :rtype: dict
    """
    resp = {'overlapped_dict': {}, 'non_over_dict': {}}
    num_cores = int(len(free_channels_dict.keys()))

    for core_num, channels_list in free_channels_dict.items():
        resp['overlapped_dict'][core_num] = list()
        resp['non_over_dict'][core_num] = list()

        for curr_channel in channels_list:
            is_overlapped = False
            for sub_core in range(0, num_cores):
                if sub_core == core_num:
                    continue

                for _, slots_dict in free_slots_dict.items():
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
                        resp['overlapped_dict'][core_num].append(curr_channel)
                        is_overlapped = True
                        break

                    resp['non_over_dict'][core_num].append(curr_channel)

                # No need to check other cores, we already determined this channel overlaps with other channels
                if is_overlapped:
                    break

    return resp


def find_free_slots(net_spec_dict: dict, link_tuple: tuple):
    """
    Find every unallocated spectral slot for a given link.

    :param net_spec_dict: The most updated network spectrum database.
    :param link_tuple: The link to find the free slots on.
    :return: The indexes of the free spectral slots on the link for each core.
    :rtype: dict
    """
    link = net_spec_dict[link_tuple]['cores_matrix']
    resp = {}
    for core_num in range(len(link)):  # pylint: disable=consider-using-enumerate
        free_slots_list = np.where(link[core_num] == 0)[0]
        resp.update({core_num: free_slots_list})

    return resp


def find_free_channels(net_spec_dict: dict, slots_needed: int, link_tuple: tuple):
    """
    Finds the free super-channels on a given link.

    :param net_spec_dict: The most updated network spectrum database.
    :param slots_needed: The number of slots needed for the request.
    :param link_tuple: The link to search on.
    :return: Available super-channels for every core.
    :rtype: dict
    """
    resp = {}
    cores_matrix = copy.deepcopy(net_spec_dict[link_tuple]['cores_matrix'])
    for core_num, link_list in enumerate(cores_matrix):
        indexes = np.where(link_list == 0)[0]
        channels_list = []
        curr_channel_list = []

        for i, free_index in enumerate(indexes):
            if i == 0:
                curr_channel_list.append(free_index)
            elif free_index == indexes[i - 1] + 1:
                curr_channel_list.append(free_index)
                if len(curr_channel_list) == slots_needed:
                    channels_list.append(curr_channel_list.copy())
                    curr_channel_list.pop(0)
            else:
                curr_channel_list = [free_index]

        resp.update({core_num: channels_list})

    return resp


def find_taken_channels(net_spec_dict: dict, link_tuple: tuple):
    """
    Finds the taken super-channels on a given link.

    :param net_spec_dict: The most updated network spectrum database.
    :param link_tuple: The link to search on.
    :return: Unavailable super-channels for every core.
    :rtype: dict
    """
    resp = {}
    cores_matrix = copy.deepcopy(net_spec_dict[link_tuple]['cores_matrix'])
    for core_num, link_list in enumerate(cores_matrix):
        channels_list = []
        curr_channel_list = []

        for value in link_list:
            if value > 0:
                curr_channel_list.append(value)
            elif value < 0 and curr_channel_list:
                channels_list.append(curr_channel_list)
                curr_channel_list = []

        if curr_channel_list:
            channels_list.append(curr_channel_list)

        resp[core_num] = channels_list

    return resp


# TODO: Doc strings, types, and optimization
def snake_to_title(snake_str: str):
    words = snake_str.split('_')
    title_str = ' '.join(word.capitalize() for word in words)
    return title_str


def int_to_string(number):
    return '{:,}'.format(number)  # pylint: disable=consider-using-f-string


def dict_to_list(data_dict, nested_key, path=None, find_mean=False):
    if path is None:
        path = []

    extracted_values = []
    for value in data_dict.values():
        for key in path:
            value = value.get(key, {})
        nested_value = value.get(nested_key)
        if nested_value is not None:
            extracted_values.append(nested_value)

    if find_mean:
        return np.mean(extracted_values)

    return np.array(extracted_values)


def list_to_title(input_list: list):
    if not input_list:
        return ""

    unique_list = list()
    for item in input_list:
        if item[0] not in unique_list:
            unique_list.append(item[0])

    if len(unique_list) > 1:
        return ", ".join(unique_list[:-1]) + " & " + unique_list[-1]

    return unique_list[0]
