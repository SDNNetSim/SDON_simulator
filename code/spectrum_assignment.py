import numpy as np


def spectrum_assignment(src_dest: tuple, slots_needed, network_spectrum_DB):
    """
    Assigns the spectrum.

    :return: The spectrum assignment (Always 1 as of now)
    :rtype: int
    """
    try:
        cores_matrix = network_spectrum_DB[src_dest]
    except KeyError:
        raise KeyError('Source to destination does not exist in the network spectrum database.')

    num_slots = np.shape(cores_matrix)[1]

    start_slot = 0
    end_slot = slots_needed - 1

    for core_num, core_arr in enumerate(cores_matrix):
        # core_arr[33:75] = 1
        # core_arr[0:] = 1
        # core_arr[0:20] = 1
        # core_arr[0:156] = 1
        core_arr[0:157] = 1

        open_slots_arr = np.where(core_arr == 0)[0]
        # Look for a super spectrum in the current core
        while end_slot < num_slots:
            spectrum_set = set(core_arr[start_slot:end_slot])
            # Spectrum is free
            if spectrum_set == {0}:
                return {'core_num': core_num, 'start_slot': start_slot, 'end_slot': end_slot}

            # No more open slots
            if len(open_slots_arr) == 0:
                break

            # TODO: This will check zero more than once
            # Jump to next available slot, assume window will shift therefore we can never pick index 0
            start_slot = open_slots_arr[0]
            # Remove prior slots
            open_slots_arr = open_slots_arr[1:]
            end_slot = start_slot + (slots_needed - 1)

    return False
