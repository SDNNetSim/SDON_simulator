import numpy as np


class SpectrumAssignment:
    """
    Finds spectrum slots for a given request.
    """

    def __init__(self, path=None, slots_needed=None, network_spec_db=None, guard_band=0, single_core=False,
                 is_sliced=False):
        self.is_free = True
        self.path = path

        self.slots_needed = slots_needed
        self.guard_band = guard_band
        self.network_spec_db = network_spec_db
        self.cores_matrix = None
        self.rev_cores_matrix = None
        self.num_slots = None
        self.single_core = single_core
        self.is_sliced = is_sliced

        self.response = {'core_num': None, 'start_slot': None, 'end_slot': None}

    def check_other_links(self, core_num, start_slot, end_slot):
        """
        Given that one link is available, check all other links in the path. Core and spectrum assignments
        MUST be the same.
        :param core_num: The core in which to look for the free spectrum
        :type core_num: int
        :param start_slot: The starting index of the potentially free spectrum
        :type start_slot: int
        :param end_slot: The ending index
        :type end_slot: int
        """
        for i, node in enumerate(self.path):  # pylint: disable=unused-variable
            if i == len(self.path) - 1:
                break
            # Ignore the first link since we check it in the method that calls this one
            if i == 0:
                continue

            # Contains source and destination names
            sub_path = (self.path[i], self.path[i + 1])
            rev_sub_path = (self.path[i + 1], self.path[i])

            spec_set = set(self.network_spec_db[sub_path]['cores_matrix'][core_num][start_slot:end_slot])
            rev_spec_set = set(self.network_spec_db[rev_sub_path]['cores_matrix'][core_num][start_slot:end_slot])

            if (spec_set, rev_spec_set) != ({0}, {0}):
                self.is_free = False
                return

            self.is_free = True

    def find_spectrum_slots(self):
        """
        Loops through each core and find the starting and ending indexes of where the request
        can be assigned.
        """
        start_slot = 0
        end_slot = self.slots_needed

        for core_num, core_arr in enumerate(self.cores_matrix):
            # To account for single core light segment slicing
            if core_num > 0 and self.single_core and self.is_sliced:
                break

            open_slots_arr = np.where(core_arr == 0)[0]

            # Look for a super channel in the current core
            while (end_slot + self.guard_band) <= self.num_slots:
                # if self.guard_band == 0 and self.slots_needed == 1:
                #     raise NotImplementedError

                spec_set = set(core_arr[start_slot:end_slot + self.guard_band])
                rev_spec_set = set(self.rev_cores_matrix[core_num][start_slot:end_slot + self.guard_band])

                # Spectrum is free
                if (spec_set, rev_spec_set) == ({0}, {0}):
                    if len(self.path) > 2:
                        self.check_other_links(core_num, start_slot, end_slot + self.guard_band)

                    # Other links spectrum slots are also available
                    if self.is_free is not False or len(self.path) <= 2:
                        self.response = {'core_num': core_num, 'start_slot': start_slot,
                                         'end_slot': end_slot + self.guard_band}
                        return

                # No more open slots
                if len(open_slots_arr) == 0:
                    self.is_free = False
                    break

                # TODO: This will check zero twice potentially (always checks zero)
                # Jump to next available slot, assume window will shift therefore we can never pick index 0
                start_slot = open_slots_arr[0]
                # Remove prior slots
                open_slots_arr = open_slots_arr[1:]
                end_slot = start_slot + self.slots_needed

            # Reset start and end slots to check the next core (if there is one)
            start_slot = 0
            end_slot = self.slots_needed

    def find_free_spectrum(self):
        """
        Controls this class.
        :return: The available core, starting index, and ending index. False otherwise.
        :rtype: dict or bool
        """
        # Ensure spectrum from 'A' to 'B' and 'B' to 'A' are free
        self.cores_matrix = self.network_spec_db[(self.path[0], self.path[1])]['cores_matrix']
        self.rev_cores_matrix = self.network_spec_db[(self.path[1], self.path[0])]['cores_matrix']

        if self.cores_matrix is None or self.rev_cores_matrix is None:
            raise ValueError('Bi-directional link not found in network spectrum database.')

        self.num_slots = np.shape(self.cores_matrix)[1]
        self.find_spectrum_slots()

        # TODO: Find a better way for this (potentially remove is_free variable)
        if self.response['start_slot'] is not None:
            return self.response

        return False