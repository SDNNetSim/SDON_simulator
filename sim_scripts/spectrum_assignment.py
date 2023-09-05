# Standard library imports
import itertools
import copy
from typing import List
from operator import itemgetter

# Third-party library imports
import numpy as np


class SpectrumAssignment:  # pylint: disable=too-few-public-methods
    """
    Finds available spectrum slots for a given request.
    """

    def __init__(self, path: List[int] = None, slots_needed: int = None, net_spec_db: dict = None,
                 guard_slots: int = None, single_core: bool = False, is_sliced: bool = False, alloc_method: str = None):
        """
        Initializes the SpectrumAssignment class.

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
        self.path = path
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

    def _handle_first_last(self, flag: str = None, des_core: int = None):
        """
        Handles either first-fit or last-fit spectrum allocation.

        :param flag: A flag to determine which allocation method we'd like to use.
        :type flag: str

        :param des_core: Determine if we'd like to force allocation onto a certain core.
        :type des_core: int
        """
        # TODO: Check if this works properly
        if des_core is not None:
            matrix = self.cores_matrix[des_core]
            start = des_core
        else:
            matrix = self.cores_matrix
            start = 0

        for core_num, core_arr in enumerate(matrix, start=start):
            # To account for ONLY single core light segment slicing
            if core_num > 0 and self.single_core and self.is_sliced:
                break

            if flag == 'first_fit':
                open_slots_arr = np.where(core_arr == 0)[0]
            elif flag == 'last_fit':
                # TODO: Ensure this works
                open_slots_arr = reversed(np.where(core_arr == 0)[0])
            else:
                raise ValueError(f'Allocation flag not recognized, expected first or last, got: {flag}')

            # Source: https://stackoverflow.com/questions/3149440/splitting-list-based-on-missing-numbers-in-a-sequence
            open_slots_matrix = [list(map(itemgetter(1), g)) for k, g in
                                 itertools.groupby(enumerate(open_slots_arr), lambda i_x: i_x[0] - i_x[1])]

            for tmp_arr in open_slots_matrix:
                if len(tmp_arr) >= (self.slots_needed + self.guard_slots):
                    for start_index in tmp_arr:
                        end_index = (start_index + self.slots_needed + self.guard_slots) - 1
                        if end_index not in tmp_arr:
                            break

                        if len(self.path) > 2:
                            self._check_other_links(core_num, start_index, end_index + self.guard_slots)

                        if self.is_free is not False or len(self.path) <= 2:
                            self.response = {'core_num': core_num, 'start_slot': start_index,
                                             'end_slot': end_index + self.guard_slots}
                            return

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

        # TODO: Add core num here!
        if self.alloc_method == 'best_fit':
            self._best_fit_allocation()
        elif self.alloc_method in ('first_fit', 'last_fit'):
            self._handle_first_last(flag=self.alloc_method)
        else:
            raise NotImplementedError(f'Expected first_fit or best_fit, got: {self.alloc_method}')

        # If the start slot is none, a request couldn't be allocated
        if self.response['start_slot'] is None:
            return False

        return self.response


    def first_fit_spectrum_allocation(self, core_num, core_arr):
        """
        Loops through each core and finds the starting and ending indexes of where the request can be assigned using the
        first-fit allocation policy.

        :return: None
        """


        open_slots_arr = np.where(core_arr == 0)[0]
        # Source: https://stackoverflow.com/questions/3149440/splitting-list-based-on-missing-numbers-in-a-sequence
        open_slots_matrix = [list(map(itemgetter(1), g)) for k, g in
                                itertools.groupby(enumerate(open_slots_arr), lambda i_x: i_x[0] - i_x[1])]

        # First fit allocation
        for tmp_arr in open_slots_matrix:
            if len(tmp_arr) >= (self.slots_needed + self.guard_slots):
                for start_index in tmp_arr:
                    # if self.guard_slots:
                    end_index = (start_index + self.slots_needed + self.guard_slots) - 1
                    # else:
                    #     end_index = start_index + self.slots_needed 
                    if end_index not in tmp_arr:
                        break

                    if len(self.path) > 2:
                        self.check_other_links(core_num, start_index, end_index + self.guard_slots)

                    if self.is_free is not False or len(self.path) <= 2:
                        self.response = {'core_num': core_num, 'start_slot': start_index,
                                            'end_slot': end_index + self.guard_slots}
                        return
    
    def last_fit_spectrum_allocation(self, core_num, core_arr):
        """
        Loops through each core and finds the starting and ending indexes of where the request can be assigned using the
        first-fit allocation policy.

        :return: None
        """


        open_slots_arr = np.where(core_arr == 0)[0]
        # Source: https://stackoverflow.com/questions/3149440/splitting-list-based-on-missing-numbers-in-a-sequence
        open_slots_matrix = [list(map(itemgetter(1), g)) for k, g in
                                itertools.groupby(enumerate(open_slots_arr), lambda i_x: i_x[0] - i_x[1])]

        # First fit allocation
        for tmp_arr in reversed(open_slots_matrix):
            if len(tmp_arr) >= (self.slots_needed + self.guard_slots):
                for start_index in tmp_arr:
                    # if self.guard_slots:
                    end_index = (start_index + self.slots_needed + self.guard_slots) - 1
                    # else:
                    #     end_index = start_index + self.slots_needed 
                    if end_index not in tmp_arr:
                        break

                    if len(self.path) > 2:
                        self.check_other_links(core_num, start_index, end_index + self.guard_slots)

                    if self.is_free is not False or len(self.path) <= 2:
                        self.response = {'core_num': core_num, 'start_slot': start_index,
                                            'end_slot': end_index + self.guard_slots}
                        return

    def _find_free_slots(self, link_num: int):
        """
        Finds the number of free channels on any given link.

        :param link_num: The link number to search for channels on.
        :type link_num: int

        :return: A matrix containing the indexes to occupied or unoccupied slots on the link.
        :rtype: dict
        """
        link = self.net_spec_db[link_num]['cores_matrix']
        final_slots = {}
        for cno in range(len(link)):
            indexes =  np.where(link[cno] == 0)[0]
            final_slots.update({cno:indexes})

        return final_slots
    
    
    def _find_free_channels(self, link_num: int, free_slots: dict):
        """
        Finds the number of free channels on any given link.

        :param link_num: The link number to search for channels on.
        :type link_num: int

        :param A matrix containing the indexes to occupied or unoccupied slots on the link.
        :type: dict
        
        :return: A matrix containing the indexes to occupied or unoccupied super channels on the link.
        :rtype: list
        """
        final_channels = {}
        for cno in free_slots:
            channels = []
            curr_channel = []
            # for i, idx in enumerate(free_slots[cno]):
            #     if i == 0:
            #         curr_channel.append(idx)
            #     elif idx == free_slots[cno][i - 1] + 1:
            #         curr_channel.append(idx)
            #         if len(curr_channel) == self.slots_needed:
            #             channels.append(curr_channel)
            #             curr_channel = []
            #     else:
            #         curr_channel = []
            
            for i, idx in enumerate(free_slots[cno]):
                for j in range(0, self.slots_needed):
                    if idx+j in free_slots[cno]:
                        curr_channel.append(idx+j)
                    else:
                        curr_channel = []
                        break
                if len(curr_channel) == self.slots_needed:
                    channels.append(curr_channel)
                    curr_channel = []
                    
                    
                    
            # Check if the last group forms a subarray
            if len(curr_channel) == self.slots_needed:
                channels.append(curr_channel)
            final_channels.update({cno: channels})

        return final_channels
    
    def dict_intersection(input_dict):
        # Convert dictionary values to sets
        sets = [set(values) for values in input_dict.values()]
        
        # Find the intersection of the sets
        intersection_set = set.intersection(*sets)
    
        return intersection_set

    def _cores_status(self):
        free_slots = {}
        free_channels = {}
        slots_intersection = {}
        channel_intersection = {}
        for source, destination in zip(self.path, self.path[1:]):
            free_slots.update({ (source, destination) : self._find_free_slots( link_num = (source, destination)) })
            free_channels.update({ (source, destination) : self._find_free_channels( link_num = (source, destination), free_slots = free_slots[(source, destination) ]) })
            for cno in free_slots[(source, destination)]:
                if cno not in slots_intersection:
                    slots_intersection.update({ cno : set(free_slots[(source, destination)][cno])})
                    channel_intersection.update({cno : free_channels[(source, destination)][cno]})
                else:
                    slots_intersection[cno] = slots_intersection[cno].intersection(free_slots[(source, destination)][cno])  
                    channel_intersection[cno] = [item for item in channel_intersection[cno] if item in free_channels[(source, destination)][cno]]
        return free_slots, slots_intersection, channel_intersection
            
    def _find_taken_channels(self):
        """
        Finds the number of taken channels on any given link.

        :return: A matrix containing the indexes to occupied or unoccupied super channels on the link.
        :rtype: list
        """
        taken_channels = {}
        for source, destination in zip(self.path, self.path[1:]): 
            taken_channels.update({(source, destination):{}})
            for cno, link in enumerate(self.net_spec_db[(source, destination)]['cores_matrix']):
                channels = []
                curr_channel = []
                #link = self.net_spec_db[(source, destination)]['cores_matrix'][cno]

                for value in link:
                    if value > 0:
                        curr_channel.append(value)
                    elif value < 0 and curr_channel:
                        channels.append(curr_channel)
                        curr_channel = []

                if curr_channel:
                    channels.append(curr_channel)
                taken_channels[(source, destination)].update({cno:channels})

        return taken_channels     
    
    
       
    def _find_overlapped_channel(self, channel_intersection, free_slots):
        """
        Finds the number of taken channels on any given link.

        :return: A matrix containing the indexes to occupied or unoccupied super channels on the link.
        :rtype: list
        """
        # TODO: counting overlapping in each link and slots
        overlapped_channels = {}
        non_overlapped_channels = {}
        for cno in channel_intersection:
            overlapped_channels.update( { cno : []} )
            non_overlapped_channels.update( { cno : []} )
            for i, slot_list in enumerate(channel_intersection[cno]):
                overlaped_cnt = 0
                for linknum in free_slots:
                    for sln in slot_list:
                        for cno2 in free_slots[linknum]:
                            if cno != cno2:
                                if cno == 6 and sln not in free_slots[linknum][cno2]:
                                    overlaped_cnt += 1
                                if cno != 6:
                                    before = 5 if cno == 0 else cno - 1
                                    after = 0 if cno == 5 else cno + 1
                                    if sln not in free_slots[linknum][before]:
                                        overlaped_cnt += 1
                                    if sln not in free_slots[linknum][after]:
                                        overlaped_cnt += 1
                                    if sln not in free_slots[linknum][6]:
                                        overlaped_cnt += 1              
                        if overlaped_cnt != 0:
                            overlapped_channels[cno].append(slot_list)
                            break
                    if overlaped_cnt != 0:
                        break
                if overlaped_cnt == 0:
                    non_overlapped_channels[cno].append(slot_list)
        return non_overlapped_channels, overlapped_channels
                    
        
        
        
                     
    def xt_aware_core_allocation(self):
        """
        Loops through each core and finds the starting and ending indexes of where the request can be assigned using the
        first-fit allocation policy.

        :return: None
        """
        no_free_slot = {}
        free_slots, slots_intersection, channel_intersection = self._cores_status()
        non_overlapped_channels, overlapped_channels = self._find_overlapped_channel(channel_intersection, free_slots)
        statistics = {}
        sorted_cores = sorted(non_overlapped_channels, key=lambda k: len(non_overlapped_channels[k]))
        if len(sorted_cores) > 1:
            if 6 in sorted_cores:
                sorted_cores.remove(6)
        return sorted_cores[0]
        # for cno in channel_intersection:
        #     statistics.update({cno:{}})
        #     if len(channel_intersection[cno]) != 0:
        #         statistics[cno].update({"overlapped":len(overlapped_channels[cno])/ len(channel_intersection[cno])})
        #         statistics[cno].update({"nonoverlapped":len(non_overlapped_channels[cno])/ len(channel_intersection[cno])})
        # taken_channels = self._find_taken_channels()
        # for core_num, core_arr in enumerate(channel_intersection):
        #     print(core_num)
    
    
    
    def xt_aware_resource_allocation(self):          
        core = self.xt_aware_core_allocation()
        core_arr = copy.deepcopy(self.net_spec_db[(self.path[0], self.path[1] )]['cores_matrix'])
        for source, destination in zip(self.path, self.path[1:]):
            if (source, destination) != (self.path[0], self.path[1]):
                core_arr = core_arr + self.net_spec_db[(source, destination)]['cores_matrix']
        if core in [0, 2, 4, 6]:
            self.first_fit_spectrum_allocation(core, core_arr)
        else:
            self.last_fit_spectrum_allocation(core, core_arr)
        return self.response