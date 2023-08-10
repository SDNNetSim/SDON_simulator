import math


def create_pt(cores_per_link: int, network_data: dict):
    """
    Generates information relevant to the physical topology of the network.

    :param cores_per_link: The number of cores in each fiber link
    :type cores_per_link: int
    :param network_data: A dictionary mapping tuples of source and destination nodes to the length of the corresponding
                        link
    :type network_data: dict

    :return: A dictionary containing information for the physical topology of the network
    :rtype: dict
    """
    # Fiber properties that apply to all fibers in a link
    fiber_properties = {
        'attenuation': 0.2 / 4.343 * math.e ** -3,
        'non_linearity': 1.3 * math.e ** -3,
        'dispersion': 16 * math.e ** -6 * (1550 * math.e ** -9) ** 2 / 2 / math.pi / 3 / math.e ** 8,
        'num_cores': cores_per_link,
        'fiber_type': 0,  # TODO: Is this always supposed to be 0? Add a comment explaining why if so.
    }

    physical_topology = {
        'nodes': {node: {'type': 'CDC'} for nodes in network_data for node in nodes},
        'links': {},
    }

    for link_num, (source_node, destination_node) in enumerate(network_data, 1):
        link_properties = {
            'fiber': fiber_properties,
            'length': network_data[(source_node, destination_node)],
            'source': source_node,
            'destination': destination_node,
        }
        physical_topology['links'][link_num] = link_properties

    return physical_topology


def create_bw_info(sim_type: str = None):
    """
    Determines the number of spectral slots needed for every modulation format in each bandwidth.

    :param sim_type: The type of simulation to perform (either 'yue' or 'arash').
    :type sim_type: str or None

    :return: The number of spectral slots needed for each bandwidth and modulation format pair.
    """
    # Check if sim_type is valid
    if sim_type not in ['yue', 'arash', None]:
        raise NotImplementedError(f"Invalid simulation type '{sim_type}'")

    bw_info = dict()

    # fill the template with maximum lengths for the given bandwidth
    if sim_type == 'yue':
        bw_info.update({
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
        })
    elif sim_type == 'arash':
        bw_info.update({
            '100': {'QPSK': {'slots_needed': 3, 'max_length': 1700}, '8-QAM': {'slots_needed': 2, 'max_length': 700},
                    '16-QAM': {'slots_needed': 2, 'max_length': 500}, '64-QAM': {'slots_needed': 1, 'max_length': 100}},
            '400': {'QPSK': {'slots_needed': 10, 'max_length': 400}, '8-QAM': {'slots_needed': 7, 'max_length': 100},
                    '16-QAM': {'slots_needed': 5, 'max_length': 100}, '64-QAM': {'slots_needed': 4, 'max_length': 0}},
        })

    return bw_info
