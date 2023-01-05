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
def create_bw_info():
    """
    Determines the number of spectral slots needed for every modulation format in each bandwidth.

    :return: The number of spectral slots needed for each bandwidth and modulation format pair
    :rtype: dict
    """
    # TODO: Note, this simulator was constructed based off of two prior research papers. For that reason, some parts of
    #   this are commented out for now. Other parts, hard coded. This will eventually be changed, but must stay this
    #   way for now.
    # Max length is in km
    bw_info = {
        # '50': {'QPSK': {'max_length': 11080}, '16-QAM': {'max_length': 4750}, '64-QAM': {'max_length': 1832}},
        '100': {'QPSK': {'max_length': 5540}, '16-QAM': {'max_length': 2375}, '64-QAM': {'max_length': 916}},
        '400': {'QPSK': {'max_length': 1385}, '16-QAM': {'max_length': 594}, '64-QAM': {'max_length': 229}},
    }

    for bw, bw_obj in bw_info.items():  # pylint: disable=invalid-name
        for mod_format, mod_obj in bw_obj.items():  # pylint: disable=unused-variable
            # Hard coded values, ignoring bw = 50 Gbps, Arash's bw assumption for number of slots needed (only using
            # 100 and 400). We don't use the max length value above in this case. Also, only one modulation format was
            # used, hence we set all the modulation values to have the same reach for ease of switching between them in
            # code. This is temporary.
            if bw == '100':
                bw_obj[mod_format]['slots_needed'] = 3  # pylint: disable=unnecessary-dict-index-lookup
            elif bw == '400':
                bw_obj[mod_format]['slots_needed'] = 10  # pylint: disable=unnecessary-dict-index-lookup
            # Yue Wang's dissertation assumption on the number of slots needed for each bandwidth and modulation format
            # bw_obj[mod_format]['slots_needed'] = math.ceil(float(bw) / self.bw_slot)

    return bw_info
