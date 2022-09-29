import numpy as np


class SpectrumAssignment:
    def __init__(self, src_dest: tuple, slots_needed, network_spectrum_DB):
        self.src_dest = src_dest
        self.slots_needed = slots_needed
        self.network_spectrum_DB = network_spectrum_DB

        self.num_slots = None

    def find_spectrum_slots(self, cores_matrix):
        start_slot = 0
        end_slot = self.slots_needed - 1

        for core_num, core_arr in enumerate(cores_matrix):
            open_slots_arr = np.where(core_arr == 0)[0]
            # Look for a super spectrum in the current core
            while end_slot < self.num_slots:
                spectrum_set = set(core_arr[start_slot:end_slot])
                # Spectrum is free
                if spectrum_set == {0}:
                    return {'core_num': core_num, 'start_slot': start_slot, 'end_slot': end_slot}

                # No more open slots
                if len(open_slots_arr) == 0:
                    break

                # TODO: This will check zero twice potentially
                # Jump to next available slot, assume window will shift therefore we can never pick index 0
                start_slot = open_slots_arr[0]
                # Remove prior slots
                open_slots_arr = open_slots_arr[1:]
                end_slot = start_slot + (self.slots_needed - 1)

        return False

    def find_src_dest(self):
        try:
            cores_matrix = self.network_spectrum_DB[self.src_dest]
        except KeyError:
            raise KeyError('Source to destination does not exist in the network spectrum database.')

        self.num_slots = np.shape(cores_matrix)[1]
        return cores_matrix

    def find_free_spectrum(self):
        cores_matrix = self.find_src_dest()
        self.find_spectrum_slots(cores_matrix=cores_matrix)
