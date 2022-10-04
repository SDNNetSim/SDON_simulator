from scripts.routing import Routing
from scripts.spectrum_assignment import SpectrumAssignment


def release(network_spec_db, path, start_slot, num_slots, core_num=0):
    """
    Releases the slots in the network spectrum database e.g. sets them back to zero.

    :param network_spec_db: The current network spectrum database
    :type network_spec_db: dict
    :param path: The shortest path computed
    :type path: list
    :param start_slot: The first slot number taken
    :type start_slot: int
    :param num_slots: The number of slots occupied
    :type num_slots: int
    :param core_num: Index of the core to be released
    :type core_num: int
    :return: The updated network spectrum database
    :rtype: dict
    """
    for i in range(len(path) - 1):
        network_spec_db[(path[i], path[i + 1])][core_num][start_slot:start_slot + num_slots] = 0

    return network_spec_db


def controller_main(src, dest, request_type, physical_topology, network_spec_db, num_slots,
                    slot_num, path=None):
    """
    Either releases spectrum from the database, assigns a spectrum, or returns False otherwise.

    :param src: The source node
    :type src: str
    :param dest: The destination node
    :type dest: str
    :param request_type: The request type e.g. Release
    :type request_type: str
    :param physical_topology: Holds all the physical topology variables
    :type physical_topology: dict
    :param network_spec_db: Holds the network spectrum information
    :type network_spec_db: dict
    :param num_slots: The number of spectrum slots needed to release or request
    :type num_slots: int
    :param slot_num: The starting slot number desired
    :type slot_num: int
    :param path: The shortest path found
    :type path: list
    """
    # TODO: Fix release
    if request_type == "Release":
        network_spec_db = release(network_spec_db=network_spec_db,
                                  path=path,
                                  start_slot=slot_num,
                                  num_slots=num_slots,
                                  )
        return network_spec_db, physical_topology

    routing_obj = Routing(source=src, destination=dest, physical_topology=physical_topology,
                          network_spec_db=network_spec_db)
    selected_path = routing_obj.least_congested_path()

    if selected_path is not False:
        spectrum_assignment = SpectrumAssignment(selected_path, num_slots, network_spec_db)
        selected_sp = spectrum_assignment.find_free_spectrum()

        if selected_sp is not False:
            ras_output = {
                'path': selected_path,
                'core_num': selected_sp['core_num'],
                'starting_NO_reserved_slot': selected_sp['start_slot'],
                'ending_NO_reserved_slot': selected_sp['end_slot'],
            }
            return ras_output, network_spec_db, physical_topology

        return False

    return False
