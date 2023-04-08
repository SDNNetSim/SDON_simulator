from itertools import groupby
from operator import itemgetter

import numpy as np


# TODO: Find a more efficient, readable, and easier way to do first-fit and best-fit
# TODO: Different allocations on a bi-directional link are not supported
# TODO: Neaten up this script (repeat code, efficiency, etc.)


class SpectrumAssignment:
    """
    Finds spectrum slots for a given request.
    """

    def __init__(self, path=None, slots_needed=None, net_spec_db=None, guard_slots=None, single_core=False,
                 is_sliced=False, alloc_method=None):
        self.is_free = True
        self.path = path

        self.slots_needed = slots_needed
        self.guard_slots = guard_slots
        self.net_spec_db = net_spec_db
        self.cores_matrix = None
        self.rev_cores_matrix = None
        self.num_slots = None
        self.num_cores = None
        self.single_core = single_core
        self.is_sliced = is_sliced
        self.alloc_method = alloc_method

        self.response = {'core_num': None, 'start_slot': None, 'end_slot': None}

    def best_fit_allocation(self):
        """
        Implements the best-fit allocation policy for spectrum assignment.
        """
        res_list = list()
        tmp_dict = dict()

        # Get all available super channels
        for i in range(len(self.path) - 1):
            src_dest = (self.path[i], self.path[i + 1])
            tmp_dict[src_dest] = dict()
            for core_num in range(self.num_cores):
                core_arr = self.net_spec_db[src_dest]['cores_matrix'][core_num]
                open_slots_arr = np.where(core_arr == 0)[0]

                # See explanation and reference for this odd syntax below
                tmp_matrix = [list(map(itemgetter(1), g)) for k, g in
                              groupby(enumerate(open_slots_arr), lambda i_x: i_x[0] - i_x[1])]
                for channel in tmp_matrix:
                    if len(channel) >= self.slots_needed:
                        res_list.append({'link': src_dest, 'core': core_num, 'channel': channel})

        # Sort the list of candidate super channels
        sorted_list = sorted(res_list, key=lambda d: len(d['channel']))
        for curr_obj in sorted_list:
            for start_index in curr_obj['channel']:
                end_index = (start_index + self.slots_needed + self.guard_slots) - 1
                if end_index not in curr_obj['channel']:
                    break

                # TODO: We must always check the first link, here this doesn't do that
                if len(self.path) > 2:
                    self.check_links(curr_obj['core'], start_index, end_index + self.guard_slots)

                if self.is_free is not False or len(self.path) <= 2:
                    self.response = {'core_num': curr_obj['core'], 'start_slot': start_index,
                                     'end_slot': end_index + self.guard_slots}
                    return

    def check_links(self, core_num, start_slot, end_slot):
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
        # TODO: Check reverse cores matrix
        for i, node in enumerate(self.path):  # pylint: disable=unused-variable
            if i == len(self.path) - 1:
                break

            # Contains source and destination names
            sub_path = (self.path[i], self.path[i + 1])
            rev_sub_path = (self.path[i + 1], self.path[i])

            spec_set = set(self.net_spec_db[sub_path]['cores_matrix'][core_num][start_slot:end_slot])
            rev_spec_set = set(self.net_spec_db[rev_sub_path]['cores_matrix'][core_num][start_slot:end_slot])

            if (spec_set, rev_spec_set) != ({0}, {0}):
                self.is_free = False
                return

            self.is_free = True

    def first_fit_allocation(self):
        """
        Loops through each core and find the starting and ending indexes of where the request
        can be assigned. First-fit allocation policy.
        """
        for core_num, core_arr in enumerate(self.cores_matrix):
            # To account for single core light segment slicing
            if core_num > 0 and self.single_core and self.is_sliced:
                break

            open_slots_arr = np.where(core_arr == 0)[0]
            # Source: https://stackoverflow.com/questions/3149440/splitting-list-based-on-missing-numbers-in-a-sequence
            open_slots_matrix = [list(map(itemgetter(1), g)) for k, g in
                                 groupby(enumerate(open_slots_arr), lambda i_x: i_x[0] - i_x[1])]

            # First fit allocation
            for tmp_arr in open_slots_matrix:
                if len(tmp_arr) >= (self.slots_needed + self.guard_slots):
                    for start_index in tmp_arr:
                        end_index = (start_index + self.slots_needed + self.guard_slots) - 1
                        if end_index not in tmp_arr:
                            break

                        if len(self.path) > 2:
                            self.check_links(core_num, start_index, end_index + self.guard_slots)

                        if self.is_free is not False or len(self.path) <= 2:
                            self.response = {'core_num': core_num, 'start_slot': start_index,
                                             'end_slot': end_index + self.guard_slots}
                            return

    def find_free_spectrum(self):
        """
        Controls this class.

        :return: The available core, starting index, and ending index. False otherwise.
        :rtype: dict or bool
        """
        # Ensure spectrum from 'A' to 'B' and 'B' to 'A' are free
        self.cores_matrix = self.net_spec_db[(self.path[0], self.path[1])]['cores_matrix']
        self.rev_cores_matrix = self.net_spec_db[(self.path[1], self.path[0])]['cores_matrix']

        if self.cores_matrix is None or self.rev_cores_matrix is None:
            raise ValueError('Bi-directional link not found in network spectrum database.')

        self.num_slots = np.shape(self.cores_matrix)[1]
        # TODO: Check this
        self.num_cores = np.shape(self.cores_matrix)[0]

        if self.alloc_method == 'best-fit':
            self.best_fit_allocation()
        elif self.alloc_method == 'first-fit':
            self.first_fit_allocation()
        else:
            raise NotImplementedError

        # If the start slot is none, a request couldn't be allocated
        if self.response['start_slot'] is not None:
            return self.response

        return False
