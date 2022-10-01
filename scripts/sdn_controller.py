from routing import routing
from spectrum_assignment import SpectrumAssignment


def release(network_spec_db, path, slot_num, num_occ_slots):
    """
    Releases the slots in the network spectrum database e.g. sets them back to zero.

    :param network_spec_db: The current network spectrum database
    :type network_spec_db: dict
    :param path: The shortest path computed
    :type path: list
    :param slot_num: The first slot number taken
    :type slot_num: int
    :param num_occ_slots: The total number of slots occupied
    :type num_occ_slots: int
    :return: The updated network spectrum database
    :rtype: dict
    """
    for cnt in range(len(path) - 1):
        for slt in range(num_occ_slots):
            network_spec_db[(path[cnt], path[cnt + 1])][slot_num + slt] = 0

    return network_spec_db


# TODO: Update return value in the docstring
def controller_main(src, dest, request_type, physical_topology, network_spec_db, slots_needed,
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
    :param slots_needed: The number of spectrum slots needed
    :type slots_needed: int
    :param slot_num: The starting slot number desired
    :type slot_num: int
    :param path: The shortest path found
    :type path: list
    """
    if request_type == "Release":
        # TODO: Factor in core# for release
        network_spec_db = release(network_spec_db=network_spec_db,
                                  path=path,
                                  slot_num=slot_num,
                                  num_occ_slots=1
                                  )
        return network_spec_db, physical_topology

    selected_path = routing(src, dest, physical_topology, network_spec_db)
    if selected_path is not False:
        # TODO: Selected spectrum will now be of type dictionary
        spectrum_assignment = SpectrumAssignment((src, dest), slots_needed, network_spec_db)
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
