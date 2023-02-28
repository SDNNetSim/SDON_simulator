import math


def create_pt(num_cores, nodes_links):
    """
    Generates information relevant to the physical topology.

    :param num_cores: The number of cores in the fiber
    :type num_cores: int
    :param nodes_links: A map of link lengths between source and destination nodes
    :type nodes_links: dict
    :return: The information for the network's physical topology
    :rtype: dict
    """
    # This may change in the future, hence creating the same dictionary for all fibers in a link right now
    # Most of this information is not used at the moment
    physical_topology = {'nodes': {}, 'links': {}}
    tmp_dict = dict()

    # TODO: Should these be exponents or Euler's constant?
    tmp_dict['attenuation'] = (0.2 / 4.343) * (math.e ** -3)
    tmp_dict['non_linearity'] = 1.3 * (math.e ** -3)
    tmp_dict['dispersion'] = (16 * math.e ** -6) * ((1550 * math.e ** -9) ** 2) / (
            2 * math.pi * 3 * math.e ** 8)
    tmp_dict['num_cores'] = num_cores
    tmp_dict['fiber_type'] = 0
    link_num = 1

    for nodes, link_len in nodes_links.items():
        source = nodes[0]
        dest = nodes[1]

        physical_topology['nodes'][source] = {'type': 'CDC'}
        physical_topology['nodes'][dest] = {'type': 'CDC'}
        physical_topology['links'][link_num] = {'fiber': tmp_dict, 'length': link_len, 'source': source,
                                                'destination': dest}
        link_num += 1

    return physical_topology


# TODO: Eventually make a config file
def create_bw_info(assume=None):
    """
    Determines the number of spectral slots needed for every modulation format in each bandwidth.

    :return: The number of spectral slots needed for each bandwidth and modulation format pair
    :rtype: dict
    """
    # Max length is in km
    if assume == 'yue':
        bw_info = {
            '25': {'QPSK': {'max_length': 22160, 'slots_needed': 1}, '16-QAM': {'max_length': 9500, 'slots_needed': 1},
                   '64-QAM': {'max_length': 3664, 'slots_needed': 1}},
            '50': {'QPSK': {'max_length': 11080, 'slots_needed': 2}, '16-QAM': {'max_length': 4750, 'slots_needed': 1},
                   '64-QAM': {'max_length': 1832, 'slots_needed': 1}},
            '100': {'QPSK': {'max_length': 5540, 'slots_needed': 4}, '16-QAM': {'max_length': 2375, 'slots_needed': 2},
                    '64-QAM': {'max_length': 916, 'slots_needed': 2}},
            '200': {'QPSK': {'max_length': 2770, 'slots_needed': 8}, '16-QAM': {'max_length': 1187, 'slots_needed': 4},
                    '64-QAM': {'max_length': 458, 'slots_needed': 3}},
            '400': {'QPSK': {'max_length': 1385, 'slots_needed': 16}, '16-QAM': {'max_length': 594, 'slots_needed': 8},
                    '64-QAM': {'max_length': 229, 'slots_needed': 6}},
        }
    elif assume == 'arash':
        bw_info = {
            '100': {'QPSK': {'slots_needed': 3}},
            '400': {'QPSK': {'slots_needed': 10}},
        }
    else:
        raise NotImplementedError

    return bw_info
