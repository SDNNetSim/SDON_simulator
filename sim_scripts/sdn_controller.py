# Standard library imports
import numpy as np

# Local application imports
from sim_scripts.routing import Routing
from sim_scripts.spectrum_assignment import SpectrumAssignment
from sim_scripts.snr_measurements import SnrMeasurements
from useful_functions.sim_functions import get_path_mod, sort_dict_keys, find_path_len


class SDNController:
    """
    Handles spectrum allocation for a request in the simulation.
    """

    def __init__(self, properties: dict = None, ai_obj: object = None):
        """
        Initializes the SDNController class.

        :param properties: Contains various simulation properties.
        :type properties: dict

        :param ai_obj: Class containing all methods related to AI
        :type ai_obj: object
        """
        self.topology_info = properties['topology_info']
        self.cores_per_link = properties['cores_per_link']
        self.sim_type = properties['sim_type']
        self.alloc_method = properties['allocation_method']
        self.route_method = properties['route_method']
        self.dynamic_lps = properties['dynamic_lps']
        self.ai_algorithm = properties['ai_algorithm']
        self.beta = properties['beta']
        self.max_segments = properties['max_segments']
        self.guard_slots = properties['guard_slots']
        self.mod_per_bw = properties['mod_per_bw']
        self.ai_obj = ai_obj
        self.spectral_slots = properties['spectral_slots']
        self.check_snr = properties['check_snr']

        # The current request id number
        self.req_id = None
        # The updated network spectrum database
        self.net_spec_db = dict()
        # Source node
        self.source = None
        # Destination node
        self.destination = None
        # The current path
        self.path = None
        # The current path modulation format
        self.path_mod = None
        # The chosen bandwidth for the current request
        self.chosen_bw = None
        # Determines if light slicing is limited to a single core or not
        self.single_core = False
        # The number of transponders used to allocate the request
        self.num_transponders = 1
        # Determines whether the block was due to distance or congestion
        self.dist_block = False
        # The physical network topology as a networkX graph
        self.topology = None
        # Class related to all things for calculating the signal-to-noise ratio
        # TODO: Might be able to identify other variables from the configuration file, for example, guard band
        # TODO: Consistent naming conventions

        # TODO: Ensure topology info updated correctly
        self.snr_obj = SnrMeasurements(properties=properties)

    def release(self):
        """
        Handles a departure event. Finds where a request was previously allocated and releases it by setting the indexes
        to all zeros.

        :return: None
        """
        for src, dest in zip(self.path, self.path[1:]):
            src_dest = (src, dest)
            dest_src = (dest, src)

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

    def allocate(self, start_slot: int, end_slot: int, core_num: int):
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
        if self.guard_slots:
            end_slot = end_slot - 1
        else:
            end_slot += 1
        for src, dest in zip(self.path, self.path[1:]):
            src_dest = (src, dest)
            dest_src = (dest, src)

            # Remember, Python list indexing is up to and NOT including!
            tmp_set = set(self.net_spec_db[src_dest]['cores_matrix'][core_num][start_slot:end_slot])
            rev_tmp_set = set(self.net_spec_db[dest_src]['cores_matrix'][core_num][start_slot:end_slot])

            if tmp_set != {0.0} or rev_tmp_set != {0.0}:
                raise BufferError("Attempted to allocate a taken spectrum.")

            self.net_spec_db[src_dest]['cores_matrix'][core_num][start_slot:end_slot] = self.req_id
            self.net_spec_db[dest_src]['cores_matrix'][core_num][start_slot:end_slot] = self.req_id

            if self.guard_slots:
                if self.net_spec_db[src_dest]['cores_matrix'][core_num][end_slot] != 0.0 or \
                        self.net_spec_db[dest_src]['cores_matrix'][core_num][end_slot] != 0.0:
                    raise BufferError("Attempted to allocate a taken spectrum.")

                self.net_spec_db[src_dest]['cores_matrix'][core_num][end_slot] = (self.req_id * -1)
                self.net_spec_db[dest_src]['cores_matrix'][core_num][end_slot] = (self.req_id * -1)

    def allocate_lps(self):
        """
        Attempts to perform light path slicing (LPS) to allocate a request.

        :return: True if LPS is successfully carried out, False otherwise
        """
        if self.chosen_bw == '25' or self.max_segments == 1:
            return False

        path_len = find_path_len(self.path, self.topology)
        # Sort the dictionary in descending order by bandwidth
        modulation_formats = sort_dict_keys(self.mod_per_bw)

        for bandwidth, modulation_dict in modulation_formats.items():
            # Cannot slice to a larger bandwidth, or slice within a bandwidth itself
            if int(bandwidth) >= int(self.chosen_bw):
                continue

            tmp_format = get_path_mod(modulation_dict, path_len)
            if tmp_format is False:
                self.dist_block = True
                continue

            num_segments = int(int(self.chosen_bw) / int(bandwidth))
            if num_segments > self.max_segments:
                break
            self.num_transponders = num_segments

            is_allocated = True
            # Check if all slices can be allocated
            for _ in range(num_segments):
                spectrum_assignment = SpectrumAssignment(path=self.path,
                                                         slots_needed=modulation_dict[tmp_format]['slots_needed'],
                                                         net_spec_db=self.net_spec_db,
                                                         guard_slots=self.guard_slots, single_core=self.single_core,
                                                         is_sliced=True, alloc_method=self.alloc_method)
                selected_spectrum = spectrum_assignment.find_free_spectrum()

                if selected_spectrum is not False:
                    self.allocate(start_slot=selected_spectrum['start_slot'], end_slot=selected_spectrum['end_slot'],
                                  core_num=selected_spectrum['core_num'])
                # Clear all previously attempted allocations
                else:
                    self.release()
                    is_allocated = False
                    self.dist_block = False
                    break

            if is_allocated:
                return {bandwidth: tmp_format}

        return False

    def allocate_dynamic_lps(self):
        """
        Attempts to perform an improved version of light path slicing (LPS) to allocate a request. An 'improved version'
        refers to allocating multiple different types of bit-rates for slicing, rather than only one.

        :return: True if advanced LPS is successfully carried out, False otherwise
        """
        resp = dict()

        if self.max_segments == 1:
            return False

        path_len = find_path_len(self.path, self.topology)
        # Sort the dictionary in descending order by bandwidth
        modulation_formats = sort_dict_keys(self.mod_per_bw)

        remaining_bw = int(self.chosen_bw)
        num_segments = 0

        for bandwidth, modulation_dict in modulation_formats.items():
            # Cannot slice to a larger bandwidth, or slice within a bandwidth itself
            if int(bandwidth) >= int(self.chosen_bw):
                continue

            tmp_format = get_path_mod(modulation_dict, path_len)
            if tmp_format is False:
                self.dist_block = True
                continue

            while True:
                spectrum_assignment = SpectrumAssignment(path=self.path,
                                                         slots_needed=modulation_dict[tmp_format]['slots_needed'],
                                                         net_spec_db=self.net_spec_db,
                                                         guard_slots=self.guard_slots, single_core=self.single_core,
                                                         is_sliced=True, alloc_method=self.alloc_method)
                selected_spectrum = spectrum_assignment.find_free_spectrum()

                if selected_spectrum is not False:
                    resp.update({bandwidth: tmp_format})

                    self.allocate(start_slot=selected_spectrum['start_slot'], end_slot=selected_spectrum['end_slot'],
                                  core_num=selected_spectrum['core_num'])

                    remaining_bw -= int(bandwidth)
                    num_segments += 1

                    if num_segments > self.max_segments:
                        self.release()
                        self.dist_block = False
                        return False
                else:
                    self.dist_block = False
                    break

                if remaining_bw == 0:
                    self.num_transponders = num_segments
                    return resp
                if int(bandwidth) > remaining_bw:
                    break

        self.release()
        return False

    def handle_lps(self):
        """
        This method attempts to perform light path slicing (LPS) or dynamic LPS to allocate a request.
        If successful, it returns a response containing the allocated path, modulation format,
        and the number of transponders used. Otherwise, it returns a tuple of False and the
        value of self.dist_block indicating whether the allocation failed due to congestion or
        a length constraint.

        :return: A tuple containing the response and the updated network database or False and self.dist_block
        """

        if not self.dynamic_lps:
            resp = self.allocate_lps()
        else:
            resp = self.allocate_dynamic_lps()

        if resp is not False:
            return {'path': self.path, 'mod_format': resp,
                    'is_sliced': True}, self.net_spec_db, self.num_transponders, self.path

        return False, self.dist_block, self.path

    def _update_snr_obj(self, spectrum: dict):
        """
        Updates parameters related to the SNR measurements class.

        :param spectrum: The spectrum chosen for the current request
        :type spectrum: dict
        """
        self.snr_obj.path = self.path
        self.snr_obj.path_mod = self.path_mod
        self.snr_obj.spectrum = spectrum
        self.snr_obj.assigned_slots = spectrum['end_slot'] - spectrum['start_slot'] + 1
        self.snr_obj.spectral_slots = self.spectral_slots
        self.snr_obj.net_spec_db = self.net_spec_db

    # TODO: Clean this up (potentially move to a different file)
    def _route(self):
        """
        For a given request, attempt to route and assign a modulation format based on a routing method flag.

        :return: A selected path and modulation format.
        :rtype: tuple
        """
        routing_obj = Routing(source=self.source, destination=self.destination,
                              topology=self.topology, net_spec_db=self.net_spec_db,
                              mod_formats=self.mod_per_bw[self.chosen_bw], bandwidth=self.chosen_bw)

        if self.route_method == 'nli_aware':
            # TODO: Constant QPSK for now
            slots_needed = self.mod_per_bw[self.chosen_bw]['QPSK']['slots_needed']
            selected_path, path_mod = routing_obj.nli_aware(slots_needed=slots_needed, beta=self.beta)
        # TODO: Add to configuration file
        elif self.route_method == 'xt_aware':
            selected_path, path_mod = routing_obj.xt_aware(beta=self.beta, xt_type='with_length')
        elif self.route_method == 'least_congested':
            selected_path = routing_obj.least_congested_path()
            # TODO: Constant QPSK for now
            path_mod = 'QPSK'
        elif self.route_method == 'shortest_path':
            selected_path, path_mod = routing_obj.shortest_path()
        elif self.route_method == 'ai':
            # Used for routing related to artificial intelligence
            selected_path = self.ai_obj.route(source=int(self.source), destination=int(self.destination))

            # TODO: Make this better
            if not selected_path:
                path_mod = None
            else:
                path_len = find_path_len(path=selected_path, topology=self.topology)
                path_mod = get_path_mod(mod_formats=self.mod_per_bw[self.chosen_bw], path_len=path_len)
        else:
            raise NotImplementedError(f'Routing method not recognized, got: {self.route_method}.')

        return selected_path, path_mod

    def handle_event(self, request_type):
        """
        Handles any event that occurs in the simulation. This is the main method in this class. Returns False if a
        request has been blocked.

        :param request_type: Whether the request is an arrival or shall be released
        :type request_type: str

        :return: The response with relevant information, network database, and physical topology
        """
        # Even if the request is blocked, we still use one transponder
        self.num_transponders = 1
        self.dist_block = False

        if request_type == "release":
            self.release()
            return self.net_spec_db

        self.path, self.path_mod = self._route()

        if self.path is not False:
            if self.path_mod is not False:
                slots_needed = self.mod_per_bw[self.chosen_bw][self.path_mod]['slots_needed']
                spectrum_assignment = SpectrumAssignment(path=self.path, slots_needed=slots_needed,
                                                         net_spec_db=self.net_spec_db, guard_slots=self.guard_slots,
                                                         is_sliced=False, alloc_method=self.alloc_method)

                spectrum = spectrum_assignment.find_free_spectrum()

                if spectrum is not False:
                    if self.check_snr:
                        self._update_snr_obj(spectrum=spectrum)
                        snr_check = self.snr_obj.check_snr()
                        if not snr_check:
                            return False, self.dist_block, self.path

                        resp = {
                            'path': self.path,
                            'mod_format': self.path_mod,
                            'is_sliced': False
                        }
                        self.allocate(spectrum['start_slot'], spectrum['end_slot'], spectrum['core_num'])
                        return resp, self.net_spec_db, self.num_transponders, self.path

                # Attempt to slice the request due to a congestion constraint
                return self.handle_lps()

            self.dist_block = True
            # Attempt to slice the request due to a reach constraint
            return self.handle_lps()

        return False, True
