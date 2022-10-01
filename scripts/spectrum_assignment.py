import numpy as np


class SpectrumAssignment:
    """
    Finds spectrum slots for a given request.
    """

    def __init__(self, src_dest: tuple = None, slots_needed=None, network_spec_db=None):
        """
        The constructor.

        :param src_dest: The source and destination of the request
        :type src_dest: tuple
        :param slots_needed: The number of spectrum slots the request needs
        :type slots_needed: int
        :param network_spec_db: The network spectrum database
        :type network_spec_db: dict
        """
        self.src_dest = src_dest
        self.slots_needed = slots_needed
        self.network_spec_db = network_spec_db

        self.num_slots = None

    def find_spectrum_slots(self, cores_matrix):
        """
        Loops through each core and find the starting and ending indexes of where the request
        can be assigned.

        :param cores_matrix: Contains the array of slots for each core
        :type cores_matrix: list
        :return: A dictionary of the free core number along with the starting and ending indexes. False otherwise.
        """
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
        """
        Determines if the source and destination exists in the network.

        :return: The cores matrix containing spectrum information.
        :rtype: list
        """
        try:
            cores_matrix = self.network_spec_db[self.src_dest]
        except KeyError as exc:
            raise KeyError('Path does not exist in the network spectrum database.') from exc

        self.num_slots = np.shape(cores_matrix)[1]
        return cores_matrix

    def find_free_spectrum(self):
        """
        Controls the methods in this class.

        :return: The available core, starting index, and ending index.
        :rtype: dict
        """
        cores_matrix = self.find_src_dest()
        return self.find_spectrum_slots(cores_matrix=cores_matrix)
