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
    tmp_dict['attenuation'] = (0.2 / 4.343) * (10 ** -3)                                                 # alpha  (1/Km)
    tmp_dict['non_linearity'] = 1.3 * (10 ** -3)                                                         # gamma 1/(W · km)
    tmp_dict['dispersion'] = (16 * 10 ** -6) * ((1550 * 10 ** -9) ** 2) / (  2 * math.pi * 3 * 10 ** 8)       # beta2 (ps/nm/km )
            
    tmp_dict['num_cores'] = num_cores
    tmp_dict['fiber_type'] = 0
    tmp_dict['span_length'] = 100                                                                       # length of span (Km)
    tmp_dict['nsp'] = 1.8                                                                               # spontaneous_emission_factor 
    tmp_dict['plank'] = 6.62607004 * 10 ** -34                                                          # plank constant
    tmp_dict['mode_coupling_co'] = 2 * 10 ** -4 #1.27 * 10 ** -5                                                       # k 
    tmp_dict["bending_radius"] = 50 * 10 ** -3                                                          # r
    tmp_dict["propagation_const"] = 4 * 10 ** 6                                                         # beta
    tmp_dict["core_pitch"] = 40 * 10 ** -6                                                              #lambda
    tmp_dict['phi'] =  {'QPSK': 1, '16-QAM': 17/25, '64-QAM': 13/21}
    
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
