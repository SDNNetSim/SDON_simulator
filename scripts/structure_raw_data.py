import configparser
import os

import pandas as pd
import numpy as np

PAIRINGS_FILE_PATH = 'data/raw/europe_network.xlsx'
LINK_LEN_FILE_PATH = 'data/raw/europe_network_distance.txt'


def map_erlang_times(network='europe'):
    """
    Map the Erlang values to the inter arrival rate and holding time from a configuration file.

    :param network: The type of network
    :type network: str
    :return: The Erlang number is mapped to the values
    :rtype: dict
    """
    response_dict = dict()

    if network == 'europe':
        hold_conf_fp = 'data/raw/europe_omnetpp_hold.ini'
        inter_conf_fp = 'data/raw/europe_omnetpp_inter.ini'
    else:
        raise NotImplementedError

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


def assign_link_lengths(node_pairings=None):
    """
    Given a text file with node numbers to link lengths, i.e., 0 1 1000, match the node numbers to node
    names.

    :param node_pairings: Node numbers to actual node names
    :return: A dictionary with a node pair (src, dest): link_length
    :rtype: dict
    """
    # The input file annoyingly does not have a consistent format, can't use dataframe
    response_dict = dict()
    with open(LINK_LEN_FILE_PATH, 'r', encoding='utf-8') as curr_f:
        for line in curr_f:
            tmp_list = line.split('\t')
            src = tmp_list[0]
            dest = tmp_list[1]
            # Remove new line, leading, and trailing white space
            link_len = tmp_list[2].strip('\n').strip()

            src_dest_tuple = (node_pairings[src], node_pairings[dest])
            response_dict[src_dest_tuple] = int(link_len)

    return response_dict


def create_node_pairs():
    """
    Takes an Excel file where node numbers are mapped to actual names and creates a dictionary of
    node number to node name pairs.

    :return: A dictionary of each node number to a node name
    :rtype: dict
    """
    data_frame = pd.read_excel(PAIRINGS_FILE_PATH)
    tmp_dict = dict()

    for row in data_frame.iterrows():
        # The first few rows are empty in Excel, we want to start node_num at zero
        for node_num, node_name in enumerate(row[1][2:]):
            tmp_dict[str(node_num)] = node_name
        # Other data in the Excel file is not relevant after the first row
        break

    return tmp_dict


def structure_data():
    """
    The main structure data function

    :return: A dictionary with a node pair (src, dest): link_length
    :rtype: dict
    """
    tmp_resp = create_node_pairs()
    return assign_link_lengths(node_pairings=tmp_resp)


def main():
    """
    Controls the program.
    """
    return structure_data()


if __name__ == '__main__':
    main()
