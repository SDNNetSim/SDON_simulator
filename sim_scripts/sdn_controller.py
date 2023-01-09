import numpy as np

from sim_scripts.routing import Routing
from sim_scripts.spectrum_assignment import SpectrumAssignment


# TODO: Objectify this code
def handle_arrive_rel(req_id, network_spec_db, path, start_slot=None, num_slots=None, guard_band=None, core_num=0,
                      req_type=None):
    """
    Releases or fills slots in the network spectrum database arrays.

    :param req_id: The request's id number
    :type req_id: int
    :param network_spec_db: The current network spectrum database
    :type network_spec_db: dict
    :param path: The shortest path computed
    :type path: list
    :param start_slot: The first slot number taken or desired
    :type start_slot: int
    :param num_slots: The number of slots occupied or to be occupied (still need to add a guard band)
    :type num_slots: int
    :param guard_band: Tells us if a guard band was used or not
    :type guard_band: bool
    :param core_num: Index of the core to be released or taken
    :type core_num: int
    :param req_type: Indicates whether it's an arrival or release
    :type req_type: str
    :return: The updated network spectrum database
    :rtype: dict
    """
    for i in range(len(path) - 1):
        src_dest = (path[i], path[i + 1])
        dest_src = (path[i + 1], path[i])

        if req_type == 'arrival':
            end_index = start_slot + num_slots
            # Remember, Python list indexing is up to and NOT including!
            network_spec_db[src_dest]['cores_matrix'][core_num][start_slot:end_index] = req_id
            network_spec_db[dest_src]['cores_matrix'][core_num][start_slot:end_index] = req_id

            # A guard band for us is a -1, as it's important to differentiate the rest of the request from it
            if guard_band:
                network_spec_db[src_dest]['cores_matrix'][core_num][end_index] = (req_id * -1)
                network_spec_db[dest_src]['cores_matrix'][core_num][end_index] = (req_id * -1)
        elif req_type == 'release':
            core_arr = network_spec_db[src_dest]['cores_matrix'][core_num]
            rev_core_arr = network_spec_db[dest_src]['cores_matrix'][core_num]
            req_indexes = np.where(core_arr == req_id)
            guard_bands = np.where(core_arr == (req_id * -1))

            for index in req_indexes:
                core_arr[index] = 0
                rev_core_arr[index] = 0
            for gb_index in guard_bands:
                core_arr[gb_index] = 0
                rev_core_arr[gb_index] = 0
        else:
            raise ValueError(f'Expected release or arrival, got {req_type}.')

    return network_spec_db


def handle_lps(req_id, path, network_spec_db, physical_topology, band_width, path_mod, mod_formats, max_lps, guard_band,
               num_cores):
    # TODO: Shall we only stick with one bandwidth?
    # TODO: Are we allowed to use 25 and 200?
    # No slicing is possible
    if band_width == '25' or max_lps == 1:
        return False

    # Obtain the length of the path
    path_len = 0
    for i in range(len(path) - 1):
        path_len += physical_topology[path[i]][path[i + 1]]['length']

    # Sort the dictionary in descending order by bandwidth
    keys_lst = [int(key) for key in mod_formats.keys()]
    keys_lst.sort(reverse=True)
    mod_formats = {str(i): mod_formats[str(i)] for i in keys_lst}

    for curr_bw, obj in mod_formats.items():
        # Cannot slice to a larger bandwidth, or slice within a bandwidth itself
        if int(curr_bw) >= int(band_width):
            continue

        # Attempt to assign a modulation format
        if obj['QPSK']['max_length'] >= path_len > obj['16-QAM']['max_length']:
            tmp_format = 'QPSK'
        elif obj['16-QAM']['max_length'] >= path_len > obj['64-QAM']['max_length']:
            tmp_format = '16-QAM'
        elif obj['64-QAM']['max_length'] >= path_len:
            tmp_format = '64-QAM'
        else:
            continue

        num_slices = int(int(band_width) / int(curr_bw))
        if num_slices > max_lps:
            break

        is_allocated = True
        # Check if all slices can be allocated
        for i in range(num_slices):
            spectrum_assignment = SpectrumAssignment(path, obj[tmp_format]['slots_needed'], network_spec_db,
                                                     guard_band=guard_band)
            selected_sp = spectrum_assignment.find_free_spectrum()
            if selected_sp is not False:
                network_spec_db = handle_arrive_rel(req_id=req_id,
                                                    network_spec_db=network_spec_db,
                                                    path=path,
                                                    start_slot=selected_sp['start_slot'],
                                                    num_slots=obj[tmp_format]['slots_needed'],
                                                    req_type='arrival',
                                                    guard_band=guard_band
                                                    )
            # Clear all previously attempted allocations
            else:
                network_spec_db = handle_arrive_rel(req_id=req_id,
                                                    network_spec_db=network_spec_db,
                                                    path=path,
                                                    req_type='release'
                                                    )
                is_allocated = False
                break
        if is_allocated:
            return network_spec_db, physical_topology, selected_sp

    return False


# TODO: We have too many arguments, neaten this up
def controller_main(req_id, src, dest, request_type, physical_topology, network_spec_db, mod_formats,
                    slot_num=None, guard_band=None, path=None, chosen_mod=None, max_lps=None, chosen_bw=None,
                    assume='arash', bw_obj=None):
    """
    Controls arrivals and departures for requests in the simulation. Return False if a request can't be allocated.

    :param req_id: The request's id number
    :type req_id: int
    :param src: Source node
    :type src: int
    :param dest: Destination node
    :type dest: int
    :param request_type: Determine if the request is an arrival or departure
    :type request_type: str
    :param physical_topology: The physical topology information
    :type physical_topology: graph
    :param network_spec_db: The network spectrum database, holding information about the network
    :type network_spec_db: dict
    :param mod_formats: Information relating to all modulation formats
    :type mod_formats: dict
    :param slot_num: The start slot number of the request to be allocated or released
    :type slot_num: int
    :param guard_band: Tells us if a guard band was used or not
    :type guard_band: int (0 or 1 as of now)
    :param path: The chosen path
    :type path: list
    :param chosen_mod: The chosen modulation format
    :param max_lps: The maximum allowed time slices
    :type max_lps: int
    :type chosen_mod: str
    :param chosen_bw: The chosen bandwidth
    :type chosen_bw: str
    :param assume: A flag to dictate whether we're using Arash or Yue's assumptions
    :type assume: str
    :return: The modulation format, core, start slot, and end slot chosen along with the network DB and topology
    :rtype: (dict, dict, dict) or bool
    """
    if request_type == "release":
        network_spec_db = handle_arrive_rel(req_id=req_id,
                                            network_spec_db=network_spec_db,
                                            path=path,
                                            req_type='release',
                                            )
        return network_spec_db, physical_topology

    routing_obj = Routing(req_id=req_id, source=src, destination=dest, physical_topology=physical_topology,
                          network_spec_db=network_spec_db, mod_formats=mod_formats, bw=chosen_bw)

    if assume == 'yue':
        selected_path, path_mod = routing_obj.shortest_path()
    elif assume == 'arash':
        selected_path = routing_obj.least_congested_path()
        path_mod = 'QPSK'
    else:
        raise NotImplementedError

    if selected_path is not False:
        if path_mod is not False:
            slots_needed = mod_formats[path_mod]['slots_needed']
            spectrum_assignment = SpectrumAssignment(selected_path, slots_needed, network_spec_db,
                                                     guard_band=guard_band)
            selected_sp = spectrum_assignment.find_free_spectrum()

            if selected_sp is not False:
                resp = {
                    'path': selected_path,
                    'mod_format': path_mod,
                    'core_num': selected_sp['core_num'],
                    'start_res_slot': selected_sp['start_slot'],
                    'end_res_slot': selected_sp['end_slot'],
                }
                network_spec_db = handle_arrive_rel(req_id=req_id,
                                                    network_spec_db=network_spec_db,
                                                    path=selected_path,
                                                    start_slot=selected_sp['start_slot'],
                                                    num_slots=slots_needed,
                                                    req_type='arrival',
                                                    guard_band=guard_band
                                                    )
                return resp, network_spec_db, physical_topology
            else:
                lps_resp = handle_lps(req_id, selected_path, network_spec_db, physical_topology, chosen_bw, path_mod,
                                      bw_obj, max_lps, guard_band, 1)
                if lps_resp is not False:
                    resp = {
                        'path': selected_path,
                        'mod_format': path_mod,
                        'core_num': lps_resp[2]['core_num'],
                        'start_res_slot': lps_resp[2]['start_slot'],
                        'end_res_slot': lps_resp[2]['end_slot'],
                    }
                    return resp, lps_resp[0], lps_resp[1]

                return False
        else:
            # TODO: Only do this if the request has been blocked? Or always?
            # TODO: Change number of cores to variable
            # TODO: Eventually move to a 'try lps' method
            lps_resp = handle_lps(req_id, selected_path, network_spec_db, physical_topology, chosen_bw, path_mod,
                                  bw_obj, max_lps, guard_band, 1)
            if lps_resp is not False:
                # TODO: This resp variable must change (Might not be needed any longer if only used for release)
                resp = {
                    'path': selected_path,
                    'mod_format': path_mod,
                    'core_num': lps_resp[2]['core_num'],
                    'start_res_slot': lps_resp[2]['start_slot'],
                    'end_res_slot': lps_resp[2]['end_slot'],
                }
                return resp, lps_resp[0], lps_resp[1]

            return False

    return False
