import configparser

import pandas as pd  # pylint: disable=import-error
import numpy as np


# TODO: Map erlang times and create node pairs are outdated functions for the time being
def map_erlang_times(network='europe'):
    """
    Map Erlang values to the inter-arrival and holding time means from a configuration file.

    :param network: The type of network topology
    :type network: str
    :return: The Erlang value mapped to holding and inter-arrival times
    :rtype: dict
    """
    response_dict = dict()

    if network == 'europe':
        hold_conf_fp = 'data/raw/europe_omnetpp_hold.ini'
        inter_conf_fp = 'data/raw/europe_omnetpp_inter.ini'
    else:
        raise NotImplementedError

    # Erlang values that exist in our configuration file
    arr_one = np.arange(10, 100, 10)
    arr_two = np.arange(100, 850, 50)
    erlang_arr = np.concatenate((arr_one, arr_two))

    hold_config_file = configparser.ConfigParser()
    hold_config_file.read(hold_conf_fp)
    inter_config_file = configparser.ConfigParser()
    inter_config_file.read(inter_conf_fp)

    for erlang in erlang_arr:
        raw_hold_value = hold_config_file[f'Config Erlang_{erlang}']['**.holding_time']
        hold_value = float(raw_hold_value.split('(')[1][:-2])

        raw_inter_value = inter_config_file[f'Config Erlang_{erlang}']['**.interarrival_time']
        inter_value = float(raw_inter_value.split('(')[1][:-2])

        response_dict[str(erlang)] = {'holding_time_mean': hold_value, 'inter_arrival_time': inter_value}

    return response_dict


def create_node_pairs():
    """
    Takes an Excel file where node numbers are mapped to names and creates a dictionary of
    node number to its corresponding name. This function is optional.

    :return: A dictionary of each node number to a node name
    :rtype: dict
    """
    data_frame = pd.read_excel('')
    tmp_dict = dict()

    for row in data_frame.iterrows():
        # The first few rows are empty in Excel in our case, we want to start node_num at zero
        for node_num, node_name in enumerate(row[1][2:]):
            tmp_dict[str(node_num)] = node_name
        # Other data in the Excel file is not relevant after the first row (In our Excel file)
        break

    return tmp_dict


def assign_link_lengths(node_pairings=None, constant_weight=False, network_fp=None):
    """
    Assign a length to every link that exists in the topology.

    :param node_pairings: Node numbers to names
    :param constant_weight: Ignore link weight for certain routing algorithms
    :param network_fp: The name of the desired network to find link weights for
    :type network_fp: str
    :return: A dictionary with a node pair (src, dest): link_length
    :rtype: dict
    """
    # The input file annoyingly does not have a consistent format, can't use dataframe
    response_dict = dict()
    with open(network_fp, 'r', encoding='utf-8') as curr_f:
        for line in curr_f:
            tmp_list = line.split('\t')
            src = tmp_list[0]
            dest = tmp_list[1]

            if constant_weight:
                link_len = '1'
            else:
                link_len = tmp_list[2].strip('\n').strip()

            if node_pairings is not None:
                src_dest_tuple = (node_pairings[src], node_pairings[dest])
                rev_tuple = (node_pairings[dest], node_pairings[src])
            # Keep numerical numbers for nodes and not names e.g. San Francisco
            else:
                src_dest_tuple = (src, dest)
                rev_tuple = (dest, src)

            if rev_tuple in response_dict.keys() or src_dest_tuple in response_dict.keys():  # pylint: disable=consider-iterating-dictionary
                continue
            response_dict[src_dest_tuple] = int(link_len)

    return response_dict


def structure_data(constant_weight=False, network=None):
    """
    The main structure data function.

    :param constant_weight: Determines if we want to set all link lengths to one or not
    :type constant_weight: bool
    :param network: The desired network name for link weights to be read in
    :type network: str
    :return: A dictionary with a node pair (src, dest): link_length
    :rtype: dict
    """
    if network == 'USNet':
        network_fp = 'data/raw/us_network.txt'
    elif network == 'NSFNet':
        network_fp = 'data/raw/nsf_network.txt'
    elif network == 'Pan-European':
        network_fp = 'data/raw/europe_network.txt'
    else:
        raise NotImplementedError(f"Unknown network name. Expected USNet, NSFNet, or Pan-European. Got: {network}")

    return assign_link_lengths(constant_weight=constant_weight, network_fp=network_fp)


def main():
    """
    Controls this script.
    """
    return structure_data()


if __name__ == '__main__':
    main()
