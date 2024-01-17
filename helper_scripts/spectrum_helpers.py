from helper_scripts.sim_helpers import find_free_channels, find_free_slots, get_channel_overlaps


# TODO: Make sure to check link and rev_link
# TODO: Can put multiple params in constructor as well
def link_has_free_spectrum(sdn_props: dict, link, rev_link, core_num, start_slot, end_slot):
    spec = sdn_props['net_spec_dict'][link]['cores_matrix'][core_num][start_slot:end_slot]
    rev_spec = sdn_props['net_spec_dict'][rev_link]['cores_matrix'][core_num][start_slot:end_slot]

    if set(spec) == {0.0} and set(rev_spec) == {0.0}:
        return True

    return False


def check_other_links(sdn_props: dict, spectrum_props: dict, core_num: int, start_index: int, end_index: int):
    spectrum_props['is_free'] = True
    for node in range(len(spectrum_props['path_list']) - 1):
        link = (spectrum_props['path_list'][node], spectrum_props['path_list'][node + 1])
        rev_link = (spectrum_props['path_list'][node + 1], spectrum_props['path_list'][node])

        if not link_has_free_spectrum(sdn_props, link, rev_link, core_num, start_index, end_index):
            spectrum_props['is_free'] = False
            return


# TODO: Break up into two functions?
def check_open_slots(sdn_props: dict, spectrum_props: dict, engine_props: dict, open_slots_matrix: list, core_num: int):
    for tmp_arr in open_slots_matrix:
        # TODO: Slots needed has not been defined
        if len(tmp_arr) >= (spectrum_props['slots_needed'] + engine_props['guard_slots']):
            for start_index in tmp_arr:
                if engine_props['allocation_method'] == 'last_fit':
                    end_index = (start_index - spectrum_props['slots_needed'] - engine_props[
                        'guard_slots']) + 1
                else:
                    end_index = (start_index + spectrum_props['slots_needed'] + engine_props[
                        'guard_slots']) - 1
                if end_index not in tmp_arr:
                    break
                else:
                    spectrum_props['is_free'] = True

                if len(spectrum_props['path_list']) > 2:
                    if engine_props['allocation_method'] == 'last_fit':
                        # Note that these are reversed since we search in decreasing order, but allocate in
                        # increasing order
                        check_other_links(sdn_props, spectrum_props, core_num, end_index,
                                          start_index + engine_props['guard_slots'])
                    else:
                        check_other_links(sdn_props, spectrum_props, core_num, start_index,
                                          end_index + engine_props['guard_slots'])

                if spectrum_props['is_free'] is not False or len(spectrum_props['path_list']) <= 2:
                    # Since we use enumeration prior and set the matrix equal to one core, the "core_num" will
                    # always be zero even if our desired core index is different, is this lazy coding? Idek
                    # fixme no forced core here
                    # TODO: What?
                    if spectrum_props['forced_core'] is not None:
                        core_num = spectrum_props['forced_core']

                    # TODO: Can make this better
                    if engine_props['allocation_method'] == 'last_fit':
                        spectrum_props['start_slot'] = end_index
                        spectrum_props['end_slot'] = start_index + engine_props['guard_slots']
                    else:
                        spectrum_props['start_slot'] = start_index
                        spectrum_props['end_slot'] = end_index + engine_props['guard_slots']

                    spectrum_props['core_num'] = core_num
                    return True

    return False


# TODO: Haven't technically used xt allocation, just make sure it runs
# TODO: Definitely to helper script
def check_cores_channels(sdn_props: dict, spectrum_props: dict):
    resp = {'free_slots': {}, 'free_channels': {}, 'slots_inters': {}, 'channel_inters': {}}

    for source_dest in zip(spectrum_props['path_list'], spectrum_props['path_list'][1:]):
        free_slots = find_free_slots(net_spec_db=sdn_props['net_spec_dict'], des_link=source_dest)
        free_channels = find_free_channels(net_spec_db=sdn_props['net_spec_dict'],
                                           slots_needed=spectrum_props['slots_needed'],
                                           des_link=source_dest)

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
    sorted_cores = sorted(all_channels['other_channels'], key=lambda k: len(all_channels['other_channels'][k]))

    # TODO: Comment why
    if len(sorted_cores) > 1:
        if 6 in sorted_cores:
            sorted_cores.remove(6)
    return sorted_cores[0]

