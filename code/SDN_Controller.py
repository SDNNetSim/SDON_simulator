from routing import routing
from spectrum_assignment import spectrum_assignment


def release(network_spectrum_DB, path, slot_NO, No_occupied_slots):
    # TODO: Ask Arash about this
    if path is not None:
        for cnt in range(len(path) - 1):
            for slt in range(No_occupied_slots):
                network_spectrum_DB[(path[cnt], path[cnt + 1])][slot_NO + slt] = 0
    return network_spectrum_DB


def controller_main(request_type,
                    Physical_topology,
                    network_spectrum_DB,
                    slot_NO,
                    path=None
                    ):
    if request_type == "Release":
        network_spectrum_DB = release(network_spectrum_DB=network_spectrum_DB,
                                      path=path,
                                      slot_NO=slot_NO,
                                      No_occupied_slots=1
                                      )
        # TODO: No ras output var assigned
        return network_spectrum_DB, Physical_topology
    else:
        selected_path = routing()
        if selected_path is not False:
            selected_sp = spectrum_assignment()
            if selected_sp is not False:
                ras_output = {
                    'path': selected_path,
                    'starting_NO_reserved_slot': selected_sp,
                }
                return ras_output, network_spectrum_DB, Physical_topology
            else:
                return False
        else:
            return False
