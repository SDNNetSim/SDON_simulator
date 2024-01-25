from helper_scripts.sim_helpers import find_free_channels, find_free_slots, get_channel_overlaps


# TODO: Convert these methods to a class
def _check_free_spectrum(sdn_props: dict, link_tuple: tuple, rev_link_tuple: tuple, core_num: int, start_index: int,
                         end_index: int):
    spectrum_set = sdn_props['net_spec_dict'][link_tuple]['cores_matrix'][core_num][start_index:end_index]
    rev_spectrum_set = sdn_props['net_spec_dict'][rev_link_tuple]['cores_matrix'][core_num][start_index:end_index]

    if set(spectrum_set) == {0.0} and set(rev_spectrum_set) == {0.0}:
        return True

    return False


def check_other_links(sdn_props: dict, spectrum_props: dict, core_num: int, start_index: int, end_index: int):
    spectrum_props['is_free'] = True
    for node in range(len(spectrum_props['path_list']) - 1):
        link_tuple = (spectrum_props['path_list'][node], spectrum_props['path_list'][node + 1])
        rev_link_tuple = (spectrum_props['path_list'][node + 1], spectrum_props['path_list'][node])

        if not _check_free_spectrum(sdn_props=sdn_props, link_tuple=link_tuple, rev_link_tuple=rev_link_tuple,
                                    core_num=core_num, start_index=start_index, end_index=end_index):
            spectrum_props['is_free'] = False
            return


def _update_spec_props(spectrum_props: dict, engine_props: dict, start_index: int, end_index: int, core_num: int):
    if spectrum_props['forced_core'] is not None:
        core_num = spectrum_props['forced_core']

    if engine_props['allocation_method'] == 'last_fit':
        spectrum_props['start_slot'] = end_index
        spectrum_props['end_slot'] = start_index + engine_props['guard_slots']
    else:
        spectrum_props['start_slot'] = start_index
        spectrum_props['end_slot'] = end_index + engine_props['guard_slots']

    spectrum_props['core_num'] = core_num
    return spectrum_props


def check_super_channels(sdn_props: dict, spectrum_props: dict, engine_props: dict, open_slots_matrix: list,
                         core_num: int):
    """
    Given a matrix of available super-channels, find one that can allocate the current request.

    :param sdn_props: Properties of the SDN controller.
    :param spectrum_props: Properties of the spectrum assignment class.
    :param engine_props: Properties of the engine class.
    :param open_slots_matrix: A matrix where each entry is an available super-channel's indexes.
    :param core_num: The core number which is currently being checked.
    :return: If the request can be successfully allocated.
    :rtype: bool
    """
    for super_channel in open_slots_matrix:
        if len(super_channel) >= (spectrum_props['slots_needed'] + engine_props['guard_slots']):
            for start_index in super_channel:
                if engine_props['allocation_method'] == 'last_fit':
                    end_index = (start_index - spectrum_props['slots_needed'] - engine_props['guard_slots']) + 1
                else:
                    end_index = (start_index + spectrum_props['slots_needed'] + engine_props['guard_slots']) - 1
                if end_index not in super_channel:
                    break
                else:
                    spectrum_props['is_free'] = True

                if len(spectrum_props['path_list']) > 2:
                    if engine_props['allocation_method'] == 'last_fit':
                        # Note that these are reversed since we search in descending, but allocate in ascending
                        check_other_links(sdn_props, spectrum_props, core_num, end_index,
                                          start_index + engine_props['guard_slots'])
                    else:
                        check_other_links(sdn_props, spectrum_props, core_num, start_index,
                                          end_index + engine_props['guard_slots'])

                if spectrum_props['is_free'] is not False or len(spectrum_props['path_list']) <= 2:
                    _update_spec_props(spectrum_props=spectrum_props, engine_props=engine_props,
                                       start_index=start_index, end_index=end_index, core_num=core_num)
                    return True

    return False


# TODO: Haven't technically used xt allocation, just make sure it runs
# TODO: Definitely to helper script
def check_cores_channels(sdn_props: dict, spectrum_props: dict):
    resp = {'free_slots': {}, 'free_channels': {}, 'slots_inters': {}, 'channel_inters': {}}

    for source_dest in zip(spectrum_props['path_list'], spectrum_props['path_list'][1:]):
        free_slots = find_free_slots(net_spec_dict=sdn_props['net_spec_dict'], link_tuple=source_dest)
        free_channels = find_free_channels(net_spec_dict=sdn_props['net_spec_dict'],
                                           slots_needed=spectrum_props['slots_needed'],
                                           link_tuple=source_dest)

        resp['free_slots'].update({source_dest: free_slots})
        resp['free_channels'].update({source_dest: free_channels})

        for core_num in resp['free_slots'][source_dest]:
            if core_num not in resp['slots_inters']:
                resp['slots_inters'].update({core_num: set(resp['free_slots'][source_dest][core_num])})

                resp['channel_inters'].update({core_num: resp['free_channels'][source_dest][core_num]})
            else:
                intersection = resp['slots_inters'][core_num] & set(resp['free_slots'][source_dest][core_num])
                resp['slots_inters'][core_num] = intersection
                resp['channel_inters'][core_num] = [item for item in resp['channel_inters'][core_num] if
                                                    item in resp['free_channels'][source_dest][core_num]]

    return resp


# TODO: probably to helper script
def find_best_core(sdn_props: dict, spectrum_props: dict):
    """
    Finds the core with the least amount of overlapping super channels for previously allocated requests.

    :return: The core with the least amount of overlapping channels.
    :rtype: int
    """
    path_info = check_cores_channels(sdn_props=sdn_props, spectrum_props=spectrum_props)
    all_channels = get_channel_overlaps(path_info['channel_inters'],
                                        path_info['free_slots'])
    sorted_cores = sorted(all_channels['non_over_dict'], key=lambda k: len(all_channels['non_over_dict'][k]))

    # TODO: Comment why
    if len(sorted_cores) > 1:
        if 6 in sorted_cores:
            sorted_cores.remove(6)
    return sorted_cores[0]
