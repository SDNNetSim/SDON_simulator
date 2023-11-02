# Standard library imports
import time
import numpy as np

# Local application imports
from sim_scripts.spectrum_assignment import SpectrumAssignment
from sim_scripts.snr_measurements import SnrMeasurements
from useful_functions.sim_functions import *  # pylint: disable=unused-wildcard-import


class SDNController:
    """
    Handles spectrum allocation for a request in the simulation.
    """

    def __init__(self, properties: dict = None):
        """
        Initializes the SDNController class.

        :param properties: Contains various simulation properties.
        :type properties: dict
        """
        self.sdn_props = properties
        self.ai_obj = None

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
        # The chosen bandwidth for the current request
        self.chosen_bw = None
        # Determines if light slicing is limited to a single core or not
        self.single_core = False
        # The number of transponders used to allocate the request
        self.num_transponders = 1
        # Determines whether the block was due to distance or congestion
        self.block_reason = False
        # The physical network topology as a networkX graph
        self.topology = None
        # Class related to all things for calculating the signal-to-noise ratio
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

            for core_num in range(self.sdn_props['cores_per_link']):
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
        if self.sdn_props['guard_slots']:
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

            if self.sdn_props['guard_slots']:
                if self.net_spec_db[src_dest]['cores_matrix'][core_num][end_slot] != 0.0 or \
                        self.net_spec_db[dest_src]['cores_matrix'][core_num][end_slot] != 0.0:
                    raise BufferError("Attempted to allocate a taken spectrum.")

                self.net_spec_db[src_dest]['cores_matrix'][core_num][end_slot] = self.req_id * -1
                self.net_spec_db[dest_src]['cores_matrix'][core_num][end_slot] = self.req_id * -1

    def allocate_lps(self):
        """
        Attempts to perform light path slicing (LPS) to allocate a request.

        :return: A dict of allocated bandwith, mod formats and cross talk cost if LPS is successfully carried out, False otherwise
        """
        if self.chosen_bw == '25' or self.sdn_props['max_segments'] == 1:
            return False

        path_len = find_path_len(self.path, self.topology)
        # Sort the dictionary in descending order by bandwidth
        modulation_formats = sort_dict_keys(self.sdn_props['mod_per_bw'])
        
        for bandwidth, modulation_dict in modulation_formats.items():
            # Cannot slice to a larger bandwidth, or slice within a bandwidth itself
            if int(bandwidth) >= int(self.chosen_bw):
                continue

            tmp_format = get_path_mod(modulation_dict, path_len)
            if tmp_format is False:
                self.block_reason = 'distance'
                continue

            num_segments = int(int(self.chosen_bw) / int(bandwidth))
            if num_segments > self.sdn_props['max_segments']:
                self.block_reason = 'max_segments'
                break
            self.num_transponders = num_segments

            is_allocated = True
            resp = {'mod_format': [], 'bw': [], 'xt_cost': [], 'spectrum': []}
            # Check if all slices can be allocated
            for _ in range(num_segments):
                spectrum, xt_cost, modulation = self._handle_spectrum_lps(mod_options=[tmp_format], lps_bw = bandwidth)

                if spectrum is not False or spectrum is not None:
                    self.allocate(start_slot=spectrum['start_slot'], end_slot=spectrum['end_slot'],
                                  core_num=spectrum['core_num'])
                    resp['xt_cost'].append(xt_cost)  
                    resp['mod_format'].append(tmp_format)
                    resp['bw'].append(bandwidth)
                    resp['spectrum'].append(spectrum)
                # Clear all previously attempted allocations
                else:
                    self.release()
                    is_allocated = False
                    self.block_reason = 'congestion'
                    break

            if is_allocated:
                return resp

        return False

    def allocate_dynamic_lps(self):
        """
        Attempts to perform an improved version of light path slicing (LPS) to allocate a request. An 'improved version'
        refers to allocating multiple different types of bit-rates for slicing, rather than only one.

        :return: True if advanced LPS is successfully carried out, False otherwise
        """
        resp = dict()

        if self.sdn_props['max_segments'] == 1:
            return False

        path_len = find_path_len(self.path, self.topology)
        # Sort the dictionary in descending order by bandwidth
        modulation_formats = sort_dict_keys(self.sdn_props['mod_per_bw'])

        remaining_bw = int(self.chosen_bw)
        num_segments = 0
        resp = {'mod_format': [], 'bw': [],'xt_cost': [], 'spectrum': []}
        for bandwidth, modulation_dict in modulation_formats.items():
            # Cannot slice to a larger bandwidth, or slice within a bandwidth itself
            if int(bandwidth) >= int(self.chosen_bw):
                continue

            tmp_format = get_path_mod(modulation_dict, path_len)
            if tmp_format is False:
                self.block_reason = 'distance'
                continue

            while True:
                spectrum, xt_cost, modulation = self._handle_spectrum_lps(mod_options=[tmp_format], lps_bw = bandwidth)

                if spectrum is not False:

                    self.allocate(start_slot=spectrum['start_slot'], end_slot=spectrum['end_slot'],
                                  core_num=spectrum['core_num'])
                    
                    resp['xt_cost'].append(xt_cost)  
                    resp['mod_format'].append(tmp_format)
                    resp['bw'].append(bandwidth)
                    resp['spectrum'].append(spectrum)
                    remaining_bw -= int(bandwidth)
                    num_segments += 1

                    if num_segments > self.sdn_props['max_segments']:
                        self.release()
                        self.block_reason = 'max_segments'
                        return False
                else:
                    self.block_reason = 'congestion'
                    break

                if remaining_bw == 0:
                    self.num_transponders = num_segments
                    return resp
                if int(bandwidth) > remaining_bw:
                    break

        self.release()
        self.block_reason = 'congestion'
        return False

    def handle_lps(self):
        """
        This method attempts to perform light path slicing (LPS) or dynamic LPS to allocate a request.
        If successful, it returns a response containing the allocated path, modulation format,
        and the number of transponders used. Otherwise, it returns a tuple of False and the
        value of self.block_reason indicating whether the allocation failed due to congestion or
        a length constraint.

        :return: A tuple containing the response and the updated network database or False and self.block_reason
        """
        if not self.sdn_props['dynamic_lps']:
            resp = self.allocate_lps()
        else:
            resp = self.allocate_dynamic_lps()

        if resp is not False:
            return {'path': self.path, 'mod_format': resp['mod_format'], 'bw': resp['bw'], 'spectrum': resp['spectrum'],
                    'xt_cost' : resp['xt_cost'], 'is_sliced': True}, self.net_spec_db, self.num_transponders, self.path

        return False, self.block_reason, self.path
        raise NotImplementedError(
            'Light path slicing is not supported at this point in time, but will be in the future.')

    def _handle_spectrum(self, mod_options: dict):
        """
        Given modulation options, iterate through them and attempt to allocate a request with one.

        :param mod_options: The modulation formats to consider.
        :type mod_options: dict

        :return: The spectrum found for allocation, false if none can be found.
        :rtype: dict
        """
        spectrum = None
        xt_cost = None
        mod_chosen = None
        for modulation in mod_options:
            if modulation is False:
                if self.sdn_props['max_segments'] > 1:
                    raise NotImplementedError

                continue

            spectrum, self.block_reason, xt_cost = get_spectrum(properties=self.sdn_props, chosen_bw=self.chosen_bw,
                                                                path=self.path,
                                                                net_spec_db=self.net_spec_db, modulation=modulation,
                                                                snr_obj=self.snr_obj,
                                                                path_mod=modulation)

            # We found a spectrum, no need to check other modulation formats
            if spectrum is not False:
                mod_chosen = modulation
                break

        return spectrum, xt_cost, mod_chosen
    
    def _handle_spectrum_lps(self, mod_options: dict, lps_bw: str):
        """
        Given modulation options, iterate through them and attempt to allocate a request with one.

        :param mod_options: The modulation formats to consider.
        :type mod_options: dict
        
        :param lps_bw: The bandwidth of sliced lightpath.
        :type lps_bw: str

        :return: The spectrum found for allocation, false if none can be found.
        :rtype: dict
        """
        spectrum = None
        xt_cost = None
        mod_chosen = None
        for modulation in mod_options:
            if modulation is False:
                if self.sdn_props['max_segments'] == 1:
                    raise NotImplementedError

                continue

            spectrum, self.block_reason, xt_cost = get_spectrum(properties=self.sdn_props, chosen_bw=lps_bw,
                                                                path=self.path,
                                                                net_spec_db=self.net_spec_db, modulation=modulation,
                                                                snr_obj=self.snr_obj,
                                                                path_mod=modulation)

            # We found a spectrum, no need to check other modulation formats
            if spectrum is not False:
                mod_chosen = modulation
                break

        return spectrum, xt_cost, mod_chosen
    
    def _handle_routing(self):
        resp = get_route(properties=self.sdn_props, source=self.source, destination=self.destination,
                         topology=self.topology, net_spec_db=self.net_spec_db, chosen_bw=self.chosen_bw,
                         ai_obj=self.ai_obj)
        return resp

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
        self.block_reason = None

        if request_type == "release":
            self.release()
            return self.net_spec_db

        start_time = time.time()
        paths, path_mods, path_weights = self._handle_routing()
        route_time = time.time() - start_time
        filtered_paths = []
        filtered_mods = []
        filtered_weights = []
        if self.sdn_props['filter_mods'] == True:
            for selectedpath, pathmod, pathlen in zip(paths, path_mods, path_weights):
                mod = filter_mod(mod_formats=self.sdn_props['mod_per_bw'][self.chosen_bw], path_len=pathlen)
                if len(mod)  != 0:
                    filtered_mods.append(mod)
                    filtered_weights.append(pathlen)
                    filtered_paths.append(selectedpath)
        else:
            filtered_paths = paths
            filtered_mods = path_mods
            filtered_weights = path_weights
        if filtered_mods == [] and self.sdn_props['max_segments'] == 1 :
            self.block_reason = 'distance'
            return False, self.block_reason, self.path
        self.path = paths[0]       
        for path, path_mod, path_weight in zip(filtered_paths, filtered_mods, filtered_weights):
            self.path = path

            # TODO: Spectrum assignment always overrides modulation format chosen when using check snr
            if path is not False:
                if self.sdn_props['check_snr'] != 'None':
                    mod_options = sort_nested_dict_vals(self.sdn_props['mod_per_bw'][self.chosen_bw],
                                                        nested_key='max_length')
                    # TODO: Change
                    for mod_ch in list(mod_options.keys()):
                        if mod_ch not in path_mod:
                            mod_options.pop(mod_ch)
                    if len(mod_options) == 0 and path == paths[-1]:
                        self.block_reason = 'distance'
                        if self.sdn_props['max_segments'] == 1:
                            return False, self.block_reason, self.path
                else:
                    if path_mod is not False:
                        mod_options = [path_mod]
                    else:
                        self.block_reason = 'distance'
                        if self.sdn_props['max_segments'] == 1:
                            return False, self.block_reason, self.path

                spectrum, xt_cost, modulation = self._handle_spectrum(mod_options=mod_options)
                # Request was blocked for this path
                if spectrum is False or spectrum is None:
                    continue

                resp = {
                    'path': self.path,
                    'mod_format': [modulation],
                    'allocated_bw_type': [self.chosen_bw],
                    'route_time': route_time,
                    'path_weight': path_weight,
                    'xt_cost': [xt_cost],
                    'is_sliced': False,
                    'spectrum': [spectrum]
                }
                self.allocate(spectrum['start_slot'], spectrum['end_slot'], spectrum['core_num'])
                return resp, self.net_spec_db, self.num_transponders, self.path

            self.block_reason = 'distance'
        if self.sdn_props['max_segments'] > 1:
            for path, path_weight in zip(paths, path_weights):
                self.path = path
                lps_resp = self.handle_lps()
                if lps_resp[0] != False:
                    resp = {
                        'path': lps_resp[0]['path'],
                        'mod_format': lps_resp[0]['mod_format'],
                        'allocated_bw_type':lps_resp[0]['bw'],
                        'route_time': route_time,
                        'path_weight': path_weight,
                        'xt_cost': lps_resp[0]['xt_cost'],
                        'is_sliced': True,
                        'spectrum': lps_resp[0]['spectrum']
                    }
                    return resp, lps_resp[1], lps_resp[2], self.path
                else:
                    self.blacblock_reason =  lps_resp[1]
                
        return False, self.block_reason, self.path
