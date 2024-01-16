import time
import numpy as np

from arg_scripts.sdn_args import empty_props
from sim_scripts.routing import Routing
from sim_scripts.spectrum_assignment import SpectrumAssignment
from sim_scripts.snr_measurements import SnrMeasurements


class SDNController:
    """
    This class contains methods to support software-defined network controller functionality.
    """

    def __init__(self, engine_props: dict):
        self.engine_props = engine_props
        self.sdn_props = empty_props

        self.ai_obj = None
        self.snr_obj = SnrMeasurements(properties=engine_props)
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

        for path_index, path_list in enumerate(self.route_obj.route_props['paths_list']):
            if path_list is not False:
                if self.route_obj.route_props['mod_formats_list'][path_index][0] is False:
                    self.sdn_props['was_routed'] = False
                    self.sdn_props['block_reason'] = 'distance'
                    return

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
                # TODO: Always one until segment slicing is implemented
                self.sdn_props['num_trans'] = 1

                self.allocate()
                return

        self.sdn_props['block_reason'] = 'distance'
        self.sdn_props['was_routed'] = False
