# Standard library imports
import itertools
import warnings
from typing import List
from operator import itemgetter

# Third-party library imports
import numpy as np

import useful_functions.sim_functions


class SpectrumAssignment:  # pylint: disable=too-few-public-methods
    """
    Finds available spectrum slots for a given request.
    """

    def __init__(self, print_warn: bool = None, path: List[int] = None, slots_needed: int = None,
                 net_spec_db: dict = None, guard_slots: int = None, single_core: bool = False,
                 is_sliced: bool = False, alloc_method: str = None, core: int = None):
        """
        Initializes the SpectrumAssignment class.

        :param print_warn: Determines if we want to print warnings or not.
        :type print_warn: bool

        :param path: A list of integers representing the path for the request.
        :type path: List[int]

        :param slots_needed: An integer representing the number of spectral slots needed to allocate the request.
        :type slots_needed: int

        :param net_spec_db: A dictionary representing the network spectrum database.
        :type net_spec_db: dict

        :param guard_slots: An integer representing the number of slots dedicated to the guard band.
        :type guard_slots: int

        :param single_core: A boolean value indicating whether we are allowed to slice to only a single core or not.
        :type single_core: bool

        :param is_sliced: A boolean value indicating whether the request is allowed to be sliced.
        :type is_sliced: bool

        :param alloc_method: A string representing the allocation policy.
        :type alloc_method: str
        """
        self.print_warn = print_warn
        self.path = path
        self.core = core
        self.single_core = single_core
        self.is_sliced = is_sliced
        self.alloc_method = alloc_method
        self.slots_needed = slots_needed
        self.guard_slots = guard_slots
        self.net_spec_db = net_spec_db

        # The flag to determine whether the request can be allocated
        self.is_free = True
        # A matrix containing the cores for each link in the network
        self.cores_matrix = None
        # The reversed version of the cores matrix
        self.rev_cores_matrix = None
        # The total number of slots per core
        self.slots_per_core = None
        # The total number of cores per link
        self.cores_per_link = None

        # The final response from this class
        self.response = {'core_num': None, 'start_slot': None, 'end_slot': None}

        self.find_free_slots = useful_functions.sim_functions.find_free_slots
        self.find_free_channels = useful_functions.sim_functions.find_free_channels
        self.get_channel_overlaps = useful_functions.sim_functions.get_channel_overlaps

    def _check_other_links(self, core_num, start_slot, end_slot):
        """
        Given that one link is available, check all other links in the path. Core and spectrum assignments
        MUST be the same.

        :param core_num: The core in which to look for the free spectrum
        :type core_num: int

        :param start_slot: The starting index of the potentially free spectrum
        :type start_slot: int

        :param end_slot: The ending index
        :type end_slot: int

        :return: None
        """
        self.is_free = True
        for i in range(len(self.path) - 1):
            link = (self.path[i], self.path[i + 1])
            rev_link = (self.path[i + 1], self.path[i])

            if not self._link_has_free_spectrum(link, core_num, start_slot, end_slot):
                self.is_free = False
                return

            if not self._link_has_free_spectrum(rev_link, core_num, start_slot, end_slot):
                self.is_free = False
                return

    def _link_has_free_spectrum(self, sub_path, core_num, start_slot, end_slot):
        """
        Check whether a link has the same spectrum assignment as the given core and whether the
        spectrum in the given range is free.

        :param sub_path: The sub-path to check
        :type sub_path: Tuple[str, str]

        :param core_num: The core in which to look for the free spectrum
        :type core_num: int

        :param start_slot: The starting index of the potentially free spectrum
        :type start_slot: int

        :param end_slot: The ending index
        :type end_slot: int

        :return: True if the spectrum is free and assigned to the given core, False otherwise
        """
        link = self.net_spec_db[sub_path]['cores_matrix'][core_num][start_slot:end_slot]
        return set(link) == {0.0}

    def _best_fit_allocation(self):
        """
        Searches for and allocates the best-fit super channel on each link along the path.

        :return: None
        """
        res_list = []

        # Get all potential super channels
        for (src, dest) in zip(self.path[:-1], self.path[1:]):
            for core_num in range(self.cores_per_link):
                core_arr = self.net_spec_db[(src, dest)]['cores_matrix'][core_num]
                open_slots_arr = np.where(core_arr == 0)[0]

                tmp_matrix = [list(map(itemgetter(1), g)) for k, g in
                              itertools.groupby(enumerate(open_slots_arr), lambda i_x: i_x[0] - i_x[1])]
                for channel in tmp_matrix:
                    if len(channel) >= self.slots_needed:
                        res_list.append({'link': (src, dest), 'core': core_num, 'channel': channel})

        # Sort the list of candidate super channels
        sorted_list = sorted(res_list, key=lambda d: len(d['channel']))

        for channel_dict in sorted_list:
            for start_index in channel_dict['channel']:
                end_index = (start_index + self.slots_needed + self.guard_slots) - 1
                if end_index not in channel_dict['channel']:
                    break

                if len(self.path) > 2:
                    self._check_other_links(channel_dict['core'], start_index, end_index + self.guard_slots)

                if self.is_free is not False or len(self.path) <= 2:
                    self.response = {'core_num': channel_dict['core'], 'start_slot': start_index,
                                     'end_slot': end_index + self.guard_slots}
                    return

    def _check_open_slots(self, open_slots_matrix: list, flag: bool, core_num: int):
        """
        Given a matrix of unallocated spectral slots, attempt to find a channel for the request.

        :param open_slots_matrix: A matrix containing each core and open spectral slots on them.
        :type open_slots_matrix: list

        :param flag: A flag to determine if we should allocate using first or last fit.
        :type flag: bool

        :param core_num: The core number in which to check other links for a potential allocation.
        :type core_num: int

        :return: If we were able to successfully allocate or not.
        :rtype: bool
        """
        for tmp_arr in open_slots_matrix:
            if len(tmp_arr) >= (self.slots_needed + self.guard_slots):
                for start_index in tmp_arr:
                    if flag == 'last_fit':
                        end_index = (start_index - self.slots_needed - self.guard_slots) + 1
                    else:
                        end_index = (start_index + self.slots_needed + self.guard_slots) - 1
                    if end_index not in tmp_arr:
                        break

                    if len(self.path) > 2:
                        if flag == 'last_fit':
                            # Note that these are reversed since we search in decreasing order, but allocate in
                            # increasing order
                            self._check_other_links(core_num, end_index, start_index + self.guard_slots)
                        else:
                            self._check_other_links(core_num, start_index, end_index + self.guard_slots)

                    if self.is_free is not False or len(self.path) <= 2:
                        # Since we use enumeration prior and set the matrix equal to one core, the "core_num" will
                        # always be zero even if our desired core index is different, is this lazy coding? Idek
                        if self.core is not None:
                            core_num = self.core

                        if flag == 'last_fit':
                            self.response = {'core_num': core_num, 'start_slot': end_index,
                                             'end_slot': start_index + self.guard_slots}
                        else:
                            self.response = {'core_num': core_num, 'start_slot': start_index,
                                             'end_slot': end_index + self.guard_slots}
                        return True

        return False

    # TODO: Write tests for this method
    def _handle_first_last(self, flag: str = None):
        """
        Handles either first-fit or last-fit spectrum allocation.

        :param flag: A flag to determine which allocation method we'd like to use.
        :type flag: str
        """
        if self.core is not None:
            matrix = [self.cores_matrix[self.core]]
            start = self.core
        else:
            matrix = self.cores_matrix
            start = 0

        for core_num, core_arr in enumerate(matrix, start=start):
            # To account for ONLY single core light segment slicing
            if core_num > 0 and self.single_core and self.is_sliced:
                break

            open_slots_arr = np.where(core_arr == 0)[0]

            # Source: https://stackoverflow.com/questions/3149440/splitting-list-based-on-missing-numbers-in-a-sequence
            if flag == 'last_fit':
                open_slots_matrix = [list(map(itemgetter(1), g))[::-1] for k, g in
                                     itertools.groupby(enumerate(open_slots_arr), lambda i_x: i_x[0] - i_x[1])]
            else:
                open_slots_matrix = [list(map(itemgetter(1), g)) for k, g in
                                     itertools.groupby(enumerate(open_slots_arr), lambda i_x: i_x[0] - i_x[1])]

            resp = self._check_open_slots(open_slots_matrix=open_slots_matrix, flag=flag, core_num=core_num)
            # We successfully found allocation on this core, no need to check the others
            if resp:
                return

    def _check_cores_channels(self):
        """
        For a given path, find the free channel intersections (between cores on that path) and the free slots for
        each link.

        :return: The free slots, free channel intersections, free slots intersections, and free channels.
        :rtype: dict
        """
        resp = {'free_slots': {}, 'free_channels': {}, 'slots_inters': {}, 'channel_inters': {}}

        for source_dest in zip(self.path, self.path[1:]):
            free_slots = self.find_free_slots(net_spec_db=self.net_spec_db, des_link=source_dest)
            free_channels = self.find_free_channels(net_spec_db=self.net_spec_db, slots_needed=self.slots_needed,
                                                    des_link=source_dest)

            resp['free_slots'].update({source_dest: free_slots})
            resp['free_channels'].update({source_dest: free_channels})

            for core_num in resp['free_slots'][source_dest]:
                if core_num not in resp['slots_inters']:
                    resp['slots_inters'].update({core_num: set(resp['free_slots'][source_dest][core_num])})

                    resp['channel_inters'].update({core_num: resp['free_channels'][source_dest][core_num]})
                else:
                    intersection = resp['slots_inters'][core_num] & set(resp['free_slots'][source_dest][core_num])
                    resp['slots_inters'][core_num] = intersection
                    resp['channel_inters'][core_num] = [item for item in resp['channel_inters'][core_num] if
                                                        item in resp['free_channels'][source_dest][core_num]]

        return resp

    def _find_best_core(self):
        """
        Finds the core with the least amount of overlapping super channels for previously allocated requests.

        :return: The core with the least amount of overlapping channels.
        :rtype: int
        """
        path_info = self._check_cores_channels()
        all_channels = self.get_channel_overlaps(path_info['channel_inters'],
                                                 path_info['free_slots'])
        sorted_cores = sorted(all_channels['other_channels'], key=lambda k: len(all_channels['other_channels'][k]))

        # TODO: Comment why
        if len(sorted_cores) > 1:
            if 6 in sorted_cores:
                sorted_cores.remove(6)
        return sorted_cores[0]

    def _xt_aware_allocation(self):
        """
        Cross-talk aware spectrum allocation. Attempts to allocate a request with the least amount of cross-talk
        interference on neighboring cores.

        :return: The information of the request if allocated, false otherwise.
        :rtype dict
        """
        if self.print_warn:
            warnings.warn('Method: xt_aware_allocation used in '
                          'spectrum_assignment that only supports 7 cores per fiber.')
        core = self._find_best_core()
        # Graph coloring for cores
        if core in [0, 2, 4, 6]:
            return self._handle_first_last(des_core=core, flag='first_fit')

        return self._handle_first_last(des_core=core, flag='last_fit')

    def find_free_spectrum(self):
        """
        Finds available spectrum to allocate a request based on the chosen allocation policy.

        :return: A dictionary with the available core, starting index, and ending index if available.
                 Otherwise, returns False.
        :rtype: dict or bool
        """
        # Ensure spectrum from 'A' to 'B' and 'B' to 'A' are free
        self.cores_matrix = self.net_spec_db[(self.path[0], self.path[1])]['cores_matrix']
        self.rev_cores_matrix = self.net_spec_db[(self.path[1], self.path[0])]['cores_matrix']

        if self.cores_matrix is None or self.rev_cores_matrix is None:
            raise ValueError('Bi-directional link not found in network spectrum database.')

        self.slots_per_core = len(self.cores_matrix[0])
        self.cores_per_link = len(self.cores_matrix)

        # TODO: Only first and last fit supported for the moment
        if self.alloc_method == 'best_fit':
            self._best_fit_allocation()
        elif self.alloc_method in ('first_fit', 'last_fit'):
            self._handle_first_last(flag=self.alloc_method)
        elif self.alloc_method == 'cross_talk_aware':
            self._xt_aware_allocation()
        else:
            raise NotImplementedError(f'Expected first_fit or best_fit, got: {self.alloc_method}')

        # If the start slot is none, a request couldn't be allocated
        if self.response['start_slot'] is None:
            return False

        return self.response
