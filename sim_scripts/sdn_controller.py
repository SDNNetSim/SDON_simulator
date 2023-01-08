from sim_scripts.routing import Routing
from sim_scripts.spectrum_assignment import SpectrumAssignment


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


def controller_main(req_id, src, dest, request_type, physical_topology, network_spec_db, mod_formats,
                    slot_num=None, guard_band=None, path=None, chosen_mod=None, chosen_bw=None, assume='arash'):
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

        return False

    return False
