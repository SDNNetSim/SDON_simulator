import math

from .data_constants import YUE_REACH_ASSUMPTIONS, ARASH_REACH_ASSUMPTIONS


def create_pt(cores_per_link: int, network_data: dict):
    """
    Generates information relevant to the physical topology of the network.

    :param cores_per_link: The number of independent cores in each optical fiber.
    :type cores_per_link: int

    :param network_data: A mapping of source and destination nodes to their lengths.
    :type network_data: dict

    :return: Information for the physical topology of the network.
    :rtype: dict
    """
    # Fiber properties that apply to all fibers in a link
    fiber_properties = {
        'attenuation': 0.2 / 4.343 * 1e-3,
        'non_linearity': 1.3e-3,
        'dispersion': (16e-6 * 1550e-9 ** 2) / (2 * math.pi * 3e8),
        'num_cores': cores_per_link,
        'fiber_type': 0,
        'bending_radius': 0.05,
        'mode_coupling_co': 4.0e-4,
        'propagation_const': 4e6,
        'core_pitch': 4e-5,
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
            'span_length': 100,
        }
        physical_topology['links'][link_num] = link_properties

    return physical_topology


def create_bw_info(sim_type: str):
    """
    Determines the number of spectral slots needed and allowed reach for every modulation format and bandwidth.

    :param sim_type: The type of simulation to perform.
    :type sim_type: str

    :return: The number of spectral slots needed and reach allowed for each bandwidth and modulation format pair.
    :rtype: dict
    """
    if sim_type == 'yue':
        return YUE_REACH_ASSUMPTIONS

    if sim_type == 'arash':
        return ARASH_REACH_ASSUMPTIONS
    raise NotImplementedError(f"Invalid simulation type expected yue or arash and got: {sim_type}")
