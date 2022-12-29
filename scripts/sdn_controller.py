from scripts.routing import Routing
from scripts.spectrum_assignment import SpectrumAssignment


def handle_arrive_rel(network_spec_db, path, start_slot, num_slots, core_num=0, req_type=None):
    """
    Releases or fills slots in the network spectrum database arrays.

    :param network_spec_db: The current network spectrum database
    :type network_spec_db: dict
    :param path: The shortest path computed
    :type path: list
    :param start_slot: The first slot number taken or desired
    :type start_slot: int
    :param num_slots: The number of slots occupied or to be occupied
    :type num_slots: int
    :param core_num: Index of the core to be released or taken
    :type core_num: int
    :param req_type: Indicates whether it's an arrival or release
    :type req_type: str
    :return: The updated network spectrum database
    :rtype: dict
    """
    if req_type == 'Arrival':
        value = 1
    elif req_type == 'Release':
        value = 0
    else:
        raise TypeError('Expected release or arrival, got None.')

    for i in range(len(path) - 1):
        network_spec_db[(path[i], path[i + 1])]['cores_matrix'][core_num][start_slot:start_slot + num_slots] = value
        network_spec_db[(path[i + 1], path[i])]['cores_matrix'][core_num][start_slot:start_slot + num_slots] = value

    return network_spec_db


def controller_main(src, dest, request_type, physical_topology, network_spec_db, mod_formats,
                    slot_num=None, path=None, chosen_mod=None, chosen_bw=None):
    if request_type == "Release":
        network_spec_db = handle_arrive_rel(network_spec_db=network_spec_db,
                                            path=path,
                                            start_slot=slot_num,
                                            num_slots=mod_formats['QPSK']['slots_needed'],
                                            req_type='Release'
                                            )
        return network_spec_db, physical_topology

    routing_obj = Routing(source=src, destination=dest, physical_topology=physical_topology,
                          network_spec_db=network_spec_db, mod_formats=mod_formats, bw=chosen_bw)
    # TODO: Update slots needed
    # selected_path, path_mod, slots_needed = routing_obj.shortest_path()
    selected_path = routing_obj.least_congested_path()

    if selected_path is not False:
        # Hard coding QPSK for now, all slots needed are actually the same (Not actually QPSK)
        # TODO: Make sure the number of slots is correct
        spectrum_assignment = SpectrumAssignment(selected_path, mod_formats['QPSK']['slots_needed'], network_spec_db)
        selected_sp = spectrum_assignment.find_free_spectrum()

        if selected_sp is not False:
            ras_output = {
                'path': selected_path,
                'mod_format': 'QPSK',
                'core_num': selected_sp['core_num'],
                'start_res_slot': selected_sp['start_slot'],
                'end_res_slot': selected_sp['end_slot'],
            }
            # TODO: Make sure the number of slots is correct
            network_spec_db = handle_arrive_rel(network_spec_db=network_spec_db,
                                                path=selected_path,
                                                start_slot=selected_sp['start_slot'],
                                                num_slots=mod_formats['QPSK']['slots_needed'],
                                                req_type='Arrival'
                                                )
            return ras_output, network_spec_db, physical_topology

        return False

    return False
