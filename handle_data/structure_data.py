# Standard library imports
import configparser
import re

# Third-party library imports
import pandas as pd


def map_erlang_times(net_name: str = 'europe'):
    """
    Map Erlang values to the inter-arrival and holding time means from a configuration file.

    :param net_name: The type of network topology
    :type net_name: str

    :return: The Erlang value mapped to holding and inter-arrival times
    :rtype: dict
    """
    response_dict = {}

    hold_conf_fp = f"data/raw/{net_name}_omnetpp_hold.ini"
    inter_conf_fp = f"data/raw/{net_name}_omnetpp_inter.ini"

    hold_config_file = configparser.ConfigParser()
    hold_config_file.read(hold_conf_fp)
    inter_config_file = configparser.ConfigParser()
    inter_config_file.read(inter_conf_fp)

    erlang_section = hold_config_file.sections()[0]
    erlang_values = hold_config_file[erlang_section]['erlangs']
    erlang_arr = list(map(int, erlang_values.split()))

    for erlang in erlang_arr:
        raw_hold_value = hold_config_file[f'Config Erlang_{erlang}']['**.holding_time']
        hold_value = float(re.search(r'\((\d+\.\d+)\)', raw_hold_value).group(1))

        raw_inter_value = inter_config_file[f'Config Erlang_{erlang}']['**.interarrival_time']
        inter_value = float(re.search(r'\((\d+\.\d+)\)', raw_inter_value).group(1))

        response_dict[str(erlang)] = {'holding_time_mean': hold_value, 'inter_arrival_time': inter_value}

    return response_dict


def create_node_pairs(excel_file: str = None):
    """
    Reads an Excel file where node numbers are mapped to names and creates a dictionary of node number to its
    corresponding name.

    :param excel_file: Path to the Excel file
    :type excel_file: str

    :return: A dictionary of each node number to a node name
    :rtype: dict
    """
    data_frame = pd.read_excel(excel_file)
    node_dict = {}
    for col in data_frame.columns[2:]:
        for node_num, node_name in enumerate(data_frame[col].values):
            if node_num == 0:
                continue
            node_dict[str(node_num - 1)] = node_name

    return node_dict


def assign_link_lengths(node_pairings: dict = None, constant_weight: bool = False, network_fp: str = None):
    """
    Assign a length to every link that exists in the topology.

    :param node_pairings: Node numbers to names
    :type node_pairings: dict

    :param constant_weight: Ignore link weight for certain routing algorithms
    :type constant_weight: bool

    :param network_fp: The name of the desired network to find link weights for
    :type network_fp: str

    :return: A dictionary with a node pair (src, dest): link_length
    :rtype: dict
    """
    response_dict = {}
    with open(network_fp, 'r', encoding='utf-8') as file_obj:
        for line in file_obj:
            src, dest, link_len_str = line.strip().split('\t')
            link_len = int(link_len_str) if not constant_weight else 1

            if node_pairings is not None:
                src_dest_tuple = (node_pairings[src], node_pairings[dest])
            else:
                src_dest_tuple = (src, dest)

            if src_dest_tuple not in response_dict:
                response_dict[src_dest_tuple] = link_len

    return response_dict


def create_network(const_weight: bool = False, net_name: str = None):
    """
    The main structure data function.

    :param const_weight: Determines if we want to set all link lengths to one or not
    :type const_weight: bool
    :param net_name: The desired network name for link weights to be read in
    :type net_name: str

    :return: A dictionary with a node pair (src, dest): link_length
    :rtype: dict
    """
    if net_name == 'USNet':
        network_fp = 'data/raw/us_network.txt'
    elif net_name == 'NSFNet':
        network_fp = 'data/raw/nsf_network.txt'
    elif net_name == 'Pan-European':
        network_fp = 'data/raw/europe_network.txt'
    elif net_name == 'Deutsche-Telekom':
        network_fp = 'data/raw/dt_network.txt'
    else:
        raise NotImplementedError(f"Unknown network name. Expected USNet, NSFNet, or Pan-European. Got: {net_name}")

    return assign_link_lengths(constant_weight=const_weight, network_fp=network_fp)
