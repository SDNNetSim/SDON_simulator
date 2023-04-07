import numpy as np

from sim_scripts.routing import Routing
from sim_scripts.spectrum_assignment import SpectrumAssignment
from useful_functions.sim_functions import get_path_mod, sort_dict_keys, find_path_len


class SDNController:
    """
    The software defined networking class. Handles events in the simulation.
    """

    def __init__(self, req_id=None, net_spec_db=None, topology=None, cores_per_link=None, path=None, sim_type=None,
                 source=None, dest=None, mod_per_bw=None, chosen_bw=None, max_slices=None, alloc_method='first-fit',
                 guard_slots=None):
        self.req_id = req_id
        self.net_spec_db = net_spec_db
        self.topology = topology
        self.cores_per_link = cores_per_link
        self.path = path
        self.sim_type = sim_type
        self.alloc_method = alloc_method

        self.source = source
        self.dest = dest

        # Modulation formats for the chosen bandwidth
        self.mod_per_bw = mod_per_bw
        self.chosen_bw = chosen_bw
        self.max_slices = max_slices
        # Limit to single core light segment slicing
        self.single_core = False
        self.transponders = 1
        self.dist_block = False

        self.guard_slots = guard_slots

    def release(self):
        """
        Handles a departure event. Finds where a request was previously allocated and releases it by setting the indexes
        to all zeros.

        :return: None
        """
        for i in range(len(self.path) - 1):
            src_dest = (self.path[i], self.path[i + 1])
            dest_src = (self.path[i + 1], self.path[i])

            for core_num in range(self.cores_per_link):
                core_arr = self.net_spec_db[src_dest]['cores_matrix'][core_num]
                req_indexes = np.where(core_arr == self.req_id)
                guard_bands = np.where(core_arr == (self.req_id * -1))

                for index in req_indexes:
                    self.net_spec_db[src_dest]['cores_matrix'][core_num][index] = 0
                    self.net_spec_db[dest_src]['cores_matrix'][core_num][index] = 0
                for gb_index in guard_bands:
                    self.net_spec_db[src_dest]['cores_matrix'][core_num][gb_index] = 0
                    self.net_spec_db[dest_src]['cores_matrix'][core_num][gb_index] = 0

    def allocate(self, start_slot, end_slot, core_num):
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
            tmp_set = set(self.net_spec_db[src_dest]['cores_matrix'][core_num][start_slot:end_slot - 1])
            rev_tmp_set = set(self.net_spec_db[dest_src]['cores_matrix'][core_num][start_slot:end_slot - 1])

            if tmp_set != {0.0} or rev_tmp_set != {0.0}:
                raise BufferError("Attempted to allocate a taken spectrum.")

            self.net_spec_db[src_dest]['cores_matrix'][core_num][start_slot:end_slot - 1] = self.req_id
            self.net_spec_db[dest_src]['cores_matrix'][core_num][start_slot:end_slot - 1] = self.req_id

            # A guard band for us is a -1, as it's important to differentiate the rest of the request from it
            if self.guard_slots:
                if self.net_spec_db[src_dest]['cores_matrix'][core_num][end_slot - 1] != 0.0 or \
                        self.net_spec_db[dest_src]['cores_matrix'][core_num][end_slot - 1] != 0.0:
                    raise BufferError("Attempted to allocate a taken spectrum.")

                self.net_spec_db[src_dest]['cores_matrix'][core_num][end_slot - 1] = (self.req_id * -1)
                self.net_spec_db[dest_src]['cores_matrix'][core_num][end_slot - 1] = (self.req_id * -1)

    def allocate_lps(self):
        """
        Attempts to perform light path slicing (lps) to allocate a request.

        :return: If we were able to successfully carry out lps or not
        """
        # Indicated whether we blocked due to congestion or a length constraint
        # No slicing is possible
        if self.chosen_bw == '25' or self.max_slices == 1:
            return False

        path_len = find_path_len(self.path, self.topology)
        # Sort the dictionary in descending order by bandwidth
        mod_formats = sort_dict_keys(self.mod_per_bw)

        for curr_bw, obj in mod_formats.items():
            # Cannot slice to a larger bandwidth, or slice within a bandwidth itself
            if int(curr_bw) >= int(self.chosen_bw):
                continue

            tmp_format = get_path_mod(obj, path_len)
            if tmp_format is False:
                self.dist_block = True
                continue

            num_slices = int(int(self.chosen_bw) / int(curr_bw))
            if num_slices > self.max_slices:
                break
            # Number of slices minus one to account for the original transponder
            self.transponders += (num_slices - 1)

            is_allocated = True
            # Check if all slices can be allocated
            for i in range(num_slices):  # pylint: disable=unused-variable
                spectrum_assignment = SpectrumAssignment(self.path, obj[tmp_format]['slots_needed'], self.net_spec_db,
                                                         guard_band=self.guard_slots, single_core=self.single_core,
                                                         is_sliced=True, allocation=self.alloc_method)
                selected_sp = spectrum_assignment.find_free_spectrum()

                if selected_sp is not False:
                    self.allocate(start_slot=selected_sp['start_slot'], end_slot=selected_sp['end_slot'],
                                  core_num=selected_sp['core_num'])
                # Clear all previously attempted allocations
                else:
                    self.release()
                    is_allocated = False
                    self.dist_block = False
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
        lps_resp = self.allocate_lps()
        if lps_resp is not False:
            resp = {
                'path': self.path,
                'mod_format': None,
                'is_sliced': True,
            }
            return resp, self.net_spec_db, self.transponders

        return False, self.dist_block

    def handle_event(self, request_type):
        """
        Handles any event that occurs in the simulation. This is the main method in this class. Returns False if a
        request has been blocked.

        :param request_type: Whether the request is an arrival or shall be released
        :type request_type: str
        :return: The response with relevant information, network database, and physical topology
        """
        # Even if the request is blocked, we still use one transponder
        self.transponders = 1
        # Whether the block is due to a distance constraint, else is a congestion constraint
        self.dist_block = False

        if request_type == "release":
            self.release()
            return self.net_spec_db

        routing_obj = Routing(source=self.source, destination=self.dest,
                              physical_topology=self.topology, network_spec_db=self.net_spec_db,
                              mod_formats=self.mod_per_bw[self.chosen_bw], bw=self.chosen_bw)

        if self.sim_type == 'yue':
            selected_path, path_mod = routing_obj.shortest_path()
        elif self.sim_type == 'arash':
            selected_path = routing_obj.least_congested_path()
            path_mod = 'QPSK'
        else:
            raise NotImplementedError

        if selected_path is not False:
            self.path = selected_path
            if path_mod is not False:
                slots_needed = self.mod_per_bw[self.chosen_bw][path_mod]['slots_needed']
                spectrum_assignment = SpectrumAssignment(self.path, slots_needed, self.net_spec_db,
                                                         guard_band=self.guard_slots, is_sliced=False,
                                                         allocation=self.alloc_method)

                selected_sp = spectrum_assignment.find_free_spectrum()

                if selected_sp is not False:
                    resp = {
                        'path': selected_path,
                        'mod_format': path_mod,
                        'is_sliced': False
                    }

                    self.allocate(selected_sp['start_slot'], selected_sp['end_slot'], selected_sp['core_num'])
                    return resp, self.net_spec_db, self.transponders

                self.dist_block = False
                return self.handle_lps()

            self.dist_block = True
            return self.handle_lps()

        raise NotImplementedError
