from sim_scripts.routing import Routing
from sim_scripts.spectrum_assignment import SpectrumAssignment


# TODO: Objectify this code
def handle_arrive_rel(req_id, network_spec_db, path, start_slot, num_slots, guard_band=None, core_num=0, req_type=None):
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
    # TODO: This will most likely change for slicing, different functions?
    for i in range(len(path) - 1):
        src_dest = (path[i], path[i + 1])

        if req_type == 'arrival':
            end_index = start_slot + num_slots
            # Remember, Python list indexing is up to and NOT including!
            network_spec_db[src_dest]['cores_matrix'][core_num][start_slot:end_index] = req_id
            # A guard band for us is a -1, as it's important to differentiate the rest of the request from it
            if guard_band:
                network_spec_db[src_dest]['cores_matrix'][core_num][end_index] = (req_id * -1)
        elif req_type == 'release':
            # To account for the guard band being released
            if guard_band:
                end_index = start_slot + num_slots + 1
            else:
                end_index = start_slot + num_slots

            network_spec_db[src_dest]['cores_matrix'][core_num][start_slot:end_index] = 0
            network_spec_db[src_dest]['cores_matrix'][core_num][start_slot:end_index] = 0
        else:
            raise ValueError(f'Expected release or arrival, got {req_type}.')

    return network_spec_db


def handle_lps(req_id, path, network_spec_db, physical_topology, band_width, path_mod, mod_formats, max_lps, guard_band,
               num_cores):
    """
        - We are basically treating them as two separate requests if we successfully split
        - After this, we'll need to change the handle arrival and release function to use np.where()
        - Restrict to single core, eventually do multiple cores
        - We need the actual path length here
    """
    orig_num_slots = mod_formats[band_width][path_mod]['slots_needed']
    # No slicing is possible
    if orig_num_slots == 1 or max_lps == 1:
        return False
    # - Iteratively check if you can split the request up between other modulation formats and bandwidths (with respect
    #   to max lps and length constraints
    # - After each successful slice, use spectrum assignment to see if the spectrum is free with fewer slots
    # - Return true if spectrum assignment returns true with the information needed
    # - Continue until slicing can no longer occur, or length does not suffice
    # - Return False if the last point happens

    path_len = 0
    for i in range(len(path) - 1):
        path_len += physical_topology[path[i]][path[i + 1]]['length']

    # Check if the length works, check if number of slots is divisible with no remainder
    # I think we want to start at the highest and work our way down to the lowest
    # Cannot slice into something that has more slots than right now
    # Damn, will have to get spectrum assignment and allocate multiple times, then release...If not then remove all?
    # Due to the above point, it would make your life a lot easier to create another method in spectrum assignment?
    # TODO: Order dict and start at selected mod, work your way down (mods and bws?)
    for curr_bw, obj in mod_formats.items():
        for modulation, obj_2 in obj.items():
            # Remember max LPS
            # TODO: Check the mod here
            if obj_2['max_length'] < path_len or obj_2['slots_needed'] % orig_num_slots != 0:
                continue

            spectrum_assignment = SpectrumAssignment(path, obj_2['slots_needed'], network_spec_db,
                                                     guard_band=guard_band)

            tmp_slots = orig_num_slots
            curr_lps = 1
            while tmp_slots != 0:
                selected_sp = spectrum_assignment.find_free_spectrum()
                # TODO: Free all spectral slots previously allocated
                if selected_sp is False:
                    # TODO: Eventually won't need start slot or number of slots
                    network_spec_db = handle_arrive_rel(req_id=req_id,
                                                        network_spec_db=network_spec_db,
                                                        path=path,
                                                        start_slot=0,
                                                        num_slots=0,
                                                        req_type='release',
                                                        guard_band=guard_band
                                                        )
                    break

                network_spec_db = handle_arrive_rel(req_id=req_id,
                                                    network_spec_db=network_spec_db,
                                                    path=path,
                                                    start_slot=selected_sp['start_slot'],
                                                    num_slots=obj_2['slots_needed'],
                                                    req_type='arrival',
                                                    guard_band=guard_band
                                                    )
                tmp_slots -= obj_2['slots_needed']
                curr_lps += 1
                if curr_lps > max_lps:
                    break
            # TODO: Update this return (also check this variable)
            if tmp_slots == 0:
                return True

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
                                            start_slot=slot_num,
                                            num_slots=mod_formats[chosen_mod]['slots_needed'],
                                            req_type='release',
                                            guard_band=guard_band
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

    if selected_path is not False and path_mod is not False:
        slots_needed = mod_formats[path_mod]['slots_needed']
        spectrum_assignment = SpectrumAssignment(selected_path, slots_needed, network_spec_db, guard_band=guard_band)
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
            # TODO: Only do this if the request has been blocked? Or always?
            # TODO: Don't forget to return false if you can't successfully do lps
            # TODO: Must the slicing be the same? For example, we can only slice w.r.t. one modulation and bandwidth?
            # TODO: I need all modulations for all bandwidths not just one
            # TODO: Change number of cores to variable
            lps_resp = handle_lps(req_id, selected_path, network_spec_db, physical_topology, chosen_bw, path_mod,
                                  bw_obj, max_lps, guard_band, 1)
            if lps_resp is not False:
                pass

            return False

    return False
