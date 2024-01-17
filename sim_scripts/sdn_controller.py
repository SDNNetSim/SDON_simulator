import time
import numpy as np

from arg_scripts.sdn_args import empty_props
from sim_scripts.routing import Routing
from sim_scripts.spectrum_assignment import SpectrumAssignment


class SDNController:
    """
    This class contains methods to support software-defined network controller functionality.
    """

    def __init__(self, engine_props: dict):
        self.engine_props = engine_props
        self.sdn_props = empty_props

        self.ai_obj = None
        self.route_obj = Routing(engine_props=self.engine_props, sdn_props=self.sdn_props)
        self.spectrum_obj = SpectrumAssignment(engine_props=self.engine_props, sdn_props=self.sdn_props)

    def release(self):
        """
        Removes a previously allocated request from the network.

        :return: None
        """
        for source, dest in zip(self.sdn_props['path_list'], self.sdn_props['path_list'][1:]):
            for core_num in range(self.engine_props['cores_per_link']):
                core_arr = self.sdn_props['net_spec_dict'][(source, dest)]['cores_matrix'][core_num]
                req_id_arr = np.where(core_arr == self.sdn_props['req_id'])
                gb_arr = np.where(core_arr == (self.sdn_props['req_id'] * -1))

                for req_index in req_id_arr:
                    self.sdn_props['net_spec_dict'][(source, dest)]['cores_matrix'][core_num][req_index] = 0
                    self.sdn_props['net_spec_dict'][(dest, source)]['cores_matrix'][core_num][req_index] = 0
                for gb_index in gb_arr:
                    self.sdn_props['net_spec_dict'][(source, dest)]['cores_matrix'][core_num][gb_index] = 0
                    self.sdn_props['net_spec_dict'][(dest, source)]['cores_matrix'][core_num][gb_index] = 0

    def _allocate_gb(self, core_matrix: list, rev_core_matrix: list, core_num: int, end_slot: int):
        if core_matrix[core_num][end_slot] != 0.0 or rev_core_matrix[core_num][end_slot] != 0.0:
            raise BufferError("Attempted to allocate a taken spectrum.")

        core_matrix[core_num][end_slot] = self.sdn_props['req_id'] * -1
        rev_core_matrix[core_num][end_slot] = self.sdn_props['req_id'] * -1

    def allocate(self):
        """
        Allocates a network request.
        """
        start_slot = self.spectrum_obj.spectrum_props['start_slot']
        end_slot = self.spectrum_obj.spectrum_props['end_slot']
        core_num = self.spectrum_obj.spectrum_props['core_num']

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

    def allocate_lps():
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
                spectrum, xt_cost, modulation = self._handle_spectrum_lps(mod_options=[tmp_format], lps_bw=bandwidth)

                if spectrum is not False and spectrum is not None:
                    self.allocate(start_slot=spectrum['start_slot'], end_slot=spectrum['end_slot'],
                                  core_num=spectrum['core_num'])

                    # TODO: Was this averaged afterwards? What happened with it
                    #   - Average what you can, otherwise not sure yet, just treat them as separate requests?
                    #   - The spectrum has to be "taken" before another request can be looked for
                    #       - For this reason, probably best to have in sdn controller
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

    def _handle_lss(self):
        raise NotImplementedError

    def handle_event(self, request_type: str):
        """
        Handles any event that occurs in the simulation, controls this class.

        :param request_type: Whether the request is an arrival or departure.
        :return: The properties of this class.
        :rtype: dict
        """
        # Even if the request is blocked, we still consider one transponder
        self.sdn_props['num_trans'] = 1

        if request_type == "release":
            self.release()
            return

        start_time = time.time()
        self.route_obj.get_route(ai_obj=self.ai_obj)
        route_time = time.time() - start_time

        #  TODO: Before we are looping through modulation formats? Why? Multiple for one path I'm assuming, routing
        segment_slicing = False
        while True:
            for path_index, path_list in enumerate(self.route_obj.route_props['paths_list']):
                if path_list is not False:
                    # TODO: I changed this from a return to a continue statement
                    if self.route_obj.route_props['mod_formats_list'][path_index][0] is False:
                        self.sdn_props['was_routed'] = False
                        self.sdn_props['block_reason'] = 'distance'
                        continue

                    # TODO: Route props should return all modulation formats to be considered, if multiple too
                    mod_format_list = self.route_obj.route_props['mod_formats_list'][path_index]
                    self.spectrum_obj.spectrum_props['path_list'] = path_list
                    self.spectrum_obj.get_spectrum(mod_format_list=mod_format_list)
                    # Request was blocked for this path
                    if self.spectrum_obj.spectrum_props['is_free'] is not True:
                        self.sdn_props['block_reason'] = 'congestion'
                        continue

                    self.sdn_props['was_routed'] = True
                    self.sdn_props['path_list'] = path_list
                    self.sdn_props['route_time'] = route_time
                    self.sdn_props['path_weight'] = self.route_obj.route_props['weights_list'][path_index]
                    self.sdn_props['spectrum_dict'] = self.spectrum_obj.spectrum_props
                    self.sdn_props['is_sliced'] = False

                    self.allocate()
                    return

            if self.engine_props['max_segments'] > 1 and self.sdn_props['bandwidth'] != '25':
                segment_slicing = True
            else:
                self.sdn_props['block_reason'] = 'distance'
                self.sdn_props['was_routed'] = False
                return
