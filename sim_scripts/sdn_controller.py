import time
import numpy as np

from arg_scripts.sdn_args import empty_props
from sim_scripts.routing import Routing
from sim_scripts.spectrum_assignment import SpectrumAssignment
from sim_scripts.snr_measurements import SnrMeasurements
from helper_scripts.sim_helpers import update_snr_obj, handle_snr


class SDNController:
    """
    This class contains methods to support software-defined network controller functionality.
    """

    def __init__(self, properties: dict = None):
        self.engine_props = properties
        self.sdn_props = empty_props

        self.ai_obj = None
        self.snr_obj = SnrMeasurements(properties=properties)
        self.route_obj = Routing(engine_props=self.engine_props, sdn_props=self.sdn_props)

    def release(self):
        """
        Removes a previously allocated request from the network.

        :return: None
        """
        for src, dest in zip(self.sdn_props['path_list'], self.sdn_props['path_list'][1:]):
            src_dest = (src, dest)
            dest_src = (dest, src)

            for core_num in range(self.engine_props['cores_per_link']):
                core_arr = self.sdn_props['net_spec_dict'][src_dest]['cores_matrix'][core_num]
                req_indexes = np.where(core_arr == self.sdn_props['req_id'])
                guard_bands = np.where(core_arr == (self.sdn_props['req_id'] * -1))

                for index in req_indexes:
                    self.sdn_props['net_spec_dict'][src_dest]['cores_matrix'][core_num][index] = 0
                    self.sdn_props['net_spec_dict'][dest_src]['cores_matrix'][core_num][index] = 0
                for gb_index in guard_bands:
                    self.sdn_props['net_spec_dict'][src_dest]['cores_matrix'][core_num][gb_index] = 0
                    self.sdn_props['net_spec_dict'][dest_src]['cores_matrix'][core_num][gb_index] = 0

    def _allocate_gb(self, core_matrix: list, rev_core_matrix: list, core_num: int, end_slot: int):
        if core_matrix[core_num][end_slot] != 0.0 or rev_core_matrix[core_num][end_slot] != 0.0:
            raise BufferError("Attempted to allocate a taken spectrum.")

        core_matrix[core_num][end_slot] = self.sdn_props['req_id'] * -1
        rev_core_matrix[core_num][end_slot] = self.sdn_props['req_id'] * -1

    def allocate(self, start_slot: int, end_slot: int, core_num: int):
        """
        Allocates a network request.

        :param start_slot: The starting spectral slot to allocate the request
        :param end_slot: The ending spectral slot to allocate the request
        :param core_num: The desired core to allocate the request
        :return: None
        """
        if self.engine_props['guard_slots']:
            end_slot = end_slot - 1
        else:
            end_slot += 1
        for link_tuple in zip(self.sdn_props['path_list'], self.sdn_props['path_list'][1:]):
            # Remember, Python list indexing is up to and NOT including!
            link_dict = self.sdn_props['net_spec_dict'][(link_tuple[0], link_tuple[1])]
            rev_link_dict = self.sdn_props['net_spec_dict'][(link_tuple[1], link_tuple[0])]

            tmp_set = set(link_dict['cores_matrix'][core_num][start_slot:end_slot])
            rev_tmp_set = set(rev_link_dict['cores_matrix'][core_num][start_slot:end_slot])
            if tmp_set != {0.0} or rev_tmp_set != {0.0}:
                raise BufferError("Attempted to allocate a taken spectrum.")

            core_matrix = link_dict['cores_matrix']
            rev_core_matrix = rev_link_dict['cores_matrix']
            core_matrix[core_num][start_slot:end_slot] = self.sdn_props['req_id']
            rev_core_matrix[core_num][start_slot:end_slot] = self.sdn_props['req_id']

            if self.engine_props['guard_slots']:
                self._allocate_gb(core_matrix=core_matrix, rev_core_matrix=rev_core_matrix, end_slot=end_slot,
                                  core_num=core_num)

    def handle_lps(self):
        raise NotImplementedError

    def __handle_spectrum(self, chosen_bw: str, path: list, net_spec_dict: dict, modulation: str,
                          snr_obj: object, path_mod: str):

        slots_needed = self.engine_props['mod_per_bw'][chosen_bw][modulation]['slots_needed']
        spectrum_assignment = SpectrumAssignment(print_warn=self.engine_props['warnings'],
                                                 path=path, slots_needed=slots_needed,
                                                 net_spec_db=self.sdn_props['net_spec_dict'],
                                                 guard_slots=self.engine_props['guard_slots'],
                                                 is_sliced=False,
                                                 alloc_method=self.engine_props['allocation_method'])

        spectrum = spectrum_assignment.find_free_spectrum()
        xt_cost = None

        if spectrum is not False:
            if self.engine_props['check_snr'] != 'None' and self.engine_props['check_snr'] is not None:
                update_snr_obj(snr_obj=snr_obj, spectrum=spectrum, path=path, path_mod=path_mod,
                               spectral_slots=self.engine_props['spectral_slots'], net_spec_db=net_spec_dict)
                snr_check, xt_cost = handle_snr(check_snr=self.engine_props['check_snr'], snr_obj=snr_obj)

                if not snr_check:
                    return False, 'xt_threshold', xt_cost

            # No reason for blocking, return spectrum and None
            return spectrum, None, xt_cost

        return False, 'congestion', xt_cost

    def _handle_spectrum(self, path: list, mod_options: dict):
        """
        Attempt to allocate a network request to a spectrum.

        :param mod_options: The modulation formats to consider.
        :return: The spectrum found for allocation, false if none could be found.
        :rtype: dict
        """
        spectrum = None
        xt_cost = None
        mod_chosen = None
        for modulation in mod_options:
            if modulation is False:
                if self.engine_props['max_segments'] > 1:
                    raise NotImplementedError

                continue

            # TODO: This most likely won't need many params now
            spectrum, self.block_reason, xt_cost = self.__handle_spectrum(chosen_bw=self.sdn_props['bandwidth'],
                                                                          path=path,
                                                                          net_spec_dict=self.sdn_props['net_spec_dict'],
                                                                          modulation=modulation,
                                                                          snr_obj=self.snr_obj,
                                                                          path_mod=modulation)

            # We found a spectrum, no need to check other modulation formats
            if spectrum is not False:
                mod_chosen = modulation
                break

        return spectrum, xt_cost, mod_chosen

    def handle_event(self, request_type: str):
        """
        Handles any event that occurs in the simulation, controls this class.

        :param request_type: Whether the request is an arrival or departure.
        :return: The response with relevant information, network database, and physical topology
        """
        # Even if the request is blocked, we still consider one transponder
        self.sdn_props['num_trans'] = 1

        if request_type == "release":
            self.release()
            return

        start_time = time.time()
        self.route_obj.get_route(ai_obj=self.ai_obj)
        route_time = time.time() - start_time

        for path_index, path_list in enumerate(self.route_obj.route_props['paths_list']):
            if path_list is not False:
                if self.route_obj.route_props['mod_formats_list'][path_index] is False:
                    self.sdn_props['was_routed'] = False
                    self.sdn_props['block_reason'] = 'distance'
                    return

                # TODO: Core was passed to spectrum because of the AI object, fix this, have ai_obj have a separate
                #   spectrum assignment to pass a core to here, have it be in spectrum assignment
                #   implement forced core another way
                mod_options = self.route_obj.route_props['mod_formats_list'][path_index]
                spectrum, xt_cost, modulation = self._handle_spectrum(mod_options=mod_options, path=path_list)
                # Request was blocked for this path
                if spectrum is False or spectrum is None:
                    self.sdn_props['block_reason'] = 'congestion'
                    continue

                self.sdn_props['was_routed'] = True
                self.sdn_props['path_list'] = path_list
                self.sdn_props['mod_format'] = modulation
                self.sdn_props['route_time'] = route_time
                self.sdn_props['path_weight'] = self.route_obj.route_props['weights_list'][path_index]
                self.sdn_props['is_sliced'] = False
                self.sdn_props['spectrum'] = spectrum
                # TODO: Always one until segment slicing is implemented
                self.sdn_props['num_trans'] = 1

                self.allocate(spectrum['start_slot'], spectrum['end_slot'], spectrum['core_num'])
                return

            self.sdn_props['block_reason'] = 'distance'

        self.sdn_props['was_routed'] = False
