import numpy as np

from sim_scripts.routing import Routing
from sim_scripts.spectrum_assignment import SpectrumAssignment
from useful_functions.sim_functions import get_path_mod, sort_dict_keys, find_path_len


class SDNController:
    """
    The software defined networking class. Handles events in the simulation.
    """

    def __init__(self, req_id=None, network_db=None, topology=None, num_cores=None, path=None, sim_assume=None,
                 src=None, dest=None, mod_formats=None, chosen_bw=None, max_lps=None):
        self.req_id = req_id
        self.network_db = network_db
        self.topology = topology
        self.num_cores = num_cores
        self.path = path
        self.sim_assume = sim_assume

        self.src = src
        self.dest = dest

        # Modulation formats for the chosen bandwidth
        # TODO: Change
        self.mod_formats = mod_formats
        self.chosen_bw = chosen_bw
        self.max_lps = max_lps

        if self.sim_assume == 'arash':
            self.guard_band = 0
        elif self.sim_assume == 'yue':
            self.guard_band = 1
        else:
            raise NotImplementedError

    def handle_release(self):
        """
        Handles a departure event. Finds where a request was previously allocated and releases it by setting the indexes
        to all zeros.

        :return: None
        """
        for i in range(len(self.path) - 1):
            src_dest = (self.path[i], self.path[i + 1])
            dest_src = (self.path[i + 1], self.path[i])

            for core_num in range(self.num_cores):
                core_arr = self.network_db[src_dest]['cores_matrix'][core_num]
                req_indexes = np.where(core_arr == self.req_id)
                guard_bands = np.where(core_arr == (self.req_id * -1))

                for index in req_indexes:
                    self.network_db[src_dest]['cores_matrix'][core_num][index] = 0
                    self.network_db[dest_src]['cores_matrix'][core_num][index] = 0
                for gb_index in guard_bands:
                    self.network_db[src_dest]['cores_matrix'][core_num][gb_index] = 0
                    self.network_db[dest_src]['cores_matrix'][core_num][gb_index] = 0

    def handle_arrival(self, start_slot, end_slot, core_num):
        """
        Handles an arrival event. Sets the allocated spectral slots equal to the request ID, the request ID is negative
        for the guard band to differentiate which slots are guard bands for future SNR calculations.

        :param start_slot: The starting spectral slot to allocate the request
        :type start_slot: int
        :param end_slot: The ending spectral slot to allocate the request
        :type end_slot: int
        :param core_num: The desired core to allocate the request
        :type core_num: int
        :return: None
        """
        for i in range(len(self.path) - 1):
            src_dest = (self.path[i], self.path[i + 1])
            dest_src = (self.path[i + 1], self.path[i])

            # Remember, Python list indexing is up to and NOT including!
            tmp_set = set(self.network_db[src_dest]['cores_matrix'][core_num][start_slot:end_slot - 1])
            rev_tmp_set = set(self.network_db[dest_src]['cores_matrix'][core_num][start_slot:end_slot - 1])

            if tmp_set != {0.0} or rev_tmp_set != {0.0}:
                raise BufferError("Attempted to allocate a taken spectrum.")

            self.network_db[src_dest]['cores_matrix'][core_num][start_slot:end_slot - 1] = self.req_id
            self.network_db[dest_src]['cores_matrix'][core_num][start_slot:end_slot - 1] = self.req_id

            # A guard band for us is a -1, as it's important to differentiate the rest of the request from it
            if self.guard_band:
                if self.network_db[src_dest]['cores_matrix'][core_num][end_slot - 1] != 0.0 or \
                        self.network_db[dest_src]['cores_matrix'][core_num][end_slot - 1] != 0.0:
                    raise BufferError("Attempted to allocate a taken spectrum.")

                self.network_db[src_dest]['cores_matrix'][core_num][end_slot - 1] = (self.req_id * -1)
                self.network_db[dest_src]['cores_matrix'][core_num][end_slot - 1] = (self.req_id * -1)

    # TODO: ...Put lps (which is actually ls *sigh*) in another script?
    def allocate_lps(self):
        """
        Attempts to perform light path slicing (lps) to allocate a request.

        :return: If we were able to successfully carry out lps or not
        """
        # TODO: Shall we only stick with one bandwidth?
        # TODO: Are we allowed to use 25 and 200?
        # No slicing is possible
        if self.chosen_bw == '25' or self.max_lps == 1:
            return False

        path_len = find_path_len(self.path, self.topology)
        # Sort the dictionary in descending order by bandwidth
        mod_formats = sort_dict_keys(self.mod_formats)

        for curr_bw, obj in mod_formats.items():
            # Cannot slice to a larger bandwidth, or slice within a bandwidth itself
            if int(curr_bw) >= int(self.chosen_bw):
                continue

            tmp_format = get_path_mod(obj, path_len)
            if tmp_format is False:
                continue

            num_slices = int(int(self.chosen_bw) / int(curr_bw))
            if num_slices > self.max_lps:
                break

            is_allocated = True
            # Check if all slices can be allocated
            for i in range(num_slices):
                spectrum_assignment = SpectrumAssignment(self.path, obj[tmp_format]['slots_needed'], self.network_db,
                                                         guard_band=self.guard_band)
                selected_sp = spectrum_assignment.find_free_spectrum()

                if selected_sp is not False:
                    self.handle_arrival(start_slot=selected_sp['start_slot'], end_slot=selected_sp['end_slot'],
                                        core_num=selected_sp['core_num'])
                # Clear all previously attempted allocations
                else:
                    self.handle_release()
                    is_allocated = False
                    break

            if is_allocated:
                return True

        return False

    def handle_lps(self):
        """
        The main light path slicing function. Created solely for the purpose to produce less lines of code in the
        handle event method.

        :return: The updated response and network database
        """
        # TODO: Only do this if the request has been blocked? Or always?
        lps_resp = self.allocate_lps()
        if lps_resp is not False:
            resp = {
                'path': self.path,
                "mod_format": None,
                "start_slot": None,
                'is_sliced': True,
            }
            return resp, self.network_db

        return False

    def handle_event(self, request_type):
        """
        Handles any event that occurs in the simulation. This is the main method in this class. Returns False if a
        request has been blocked.

        :param request_type: Whether the request is an arrival or shall be released
        :type request_type: str
        :return: The response with relevant information, network database, and physical topology
        """
        if request_type == "release":
            self.handle_release()
            return self.network_db

        routing_obj = Routing(req_id=self.req_id, source=self.src, destination=self.dest,
                              physical_topology=self.topology, network_spec_db=self.network_db,
                              mod_formats=self.mod_formats[self.chosen_bw], bw=self.chosen_bw)

        if self.sim_assume == 'yue':
            selected_path, path_mod = routing_obj.shortest_path()
        elif self.sim_assume == 'arash':
            selected_path = routing_obj.least_congested_path()
            path_mod = 'QPSK'
        else:
            raise NotImplementedError

        if selected_path is not False:
            self.path = selected_path
            if path_mod is not False:
                slots_needed = self.mod_formats[self.chosen_bw][path_mod]['slots_needed']
                spectrum_assignment = SpectrumAssignment(self.path, slots_needed, self.network_db,
                                                         guard_band=self.guard_band)

                # TODO: Ensure spectrum assignment works correctly
                selected_sp = spectrum_assignment.find_free_spectrum()

                # TODO: Response needs to be updated (we don't need start slot and things like that anymore)
                if selected_sp is not False:
                    resp = {
                        'path': selected_path,
                        'mod_format': path_mod,
                        'core_num': selected_sp['core_num'],
                        'start_slot': selected_sp['start_slot'],
                        'end_slot': selected_sp['end_slot'],
                        'is_sliced': False
                    }

                    # TODO: We assume spectrum assignment work correctly here
                    # TODO: Debug to ensure passing the end slot works correctly
                    self.handle_arrival(selected_sp['start_slot'], selected_sp['end_slot'], selected_sp['core_num'])
                    return resp, self.network_db
                else:
                    return self.handle_lps()
            else:
                return self.handle_lps()

        return False
