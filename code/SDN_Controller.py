from routing import routing
from spectrum_assignment import spectrum_assignment


# TODO: Is slot_NO the starting slot number taken?
def release(network_spectrum_DB, path, slot_NO, No_occupied_slots):
    """
    Releases the slots in the network spectrum database e.g. sets them back to zero.

    :param network_spectrum_DB: The current network spectrum database
    :type network_spectrum_DB: dict
    :param path: The shortest path computed
    :type path: list
    :param slot_NO: The first slot number taken
    :type slot_NO: int
    :param No_occupied_slots: The total number of slots occupied
    :type No_occupied_slots: int
    :return: The updated network spectrum database
    :rtype: dict
    """
    # TODO: Ask Arash about this (If path is not None)
    if path is not None:
        for cnt in range(len(path) - 1):
            for slt in range(No_occupied_slots):
                network_spectrum_DB[(path[cnt], path[cnt + 1])][slot_NO + slt] = 0

    return network_spectrum_DB


# TODO: Update return value in the docstring
def controller_main(request_type, Physical_topology, network_spectrum_DB, slot_NO, path=None):
    """
    Either releases spectrum from the database, assigns a spectrum, or returns False otherwise.

    :param request_type: The request type e.g. Release
    :type request_type: str
    :param Physical_topology: Holds all the physical topology variables
    :type Physical_topology: dict
    :param network_spectrum_DB: Holds the network spectrum information
    :type network_spectrum_DB: dict
    :param slot_NO: The starting slot number desired
    :type slot_NO: int
    :param path: The shortest path found
    :type path: list
    """
    if request_type == "Release":
        network_spectrum_DB = release(network_spectrum_DB=network_spectrum_DB,
                                      path=path,
                                      slot_NO=slot_NO,
                                      No_occupied_slots=1
                                      )
        # TODO: No ras output var assigned
        return network_spectrum_DB, Physical_topology

    # TODO: No arguments given
    selected_path = routing()  # pylint: disable=no-value-for-parameter
    if selected_path is not False:
        selected_sp = spectrum_assignment()
        if selected_sp is not False:
            ras_output = {
                'path': selected_path,
                'starting_NO_reserved_slot': selected_sp,
            }
            return ras_output, network_spectrum_DB, Physical_topology

        return False

    return False
