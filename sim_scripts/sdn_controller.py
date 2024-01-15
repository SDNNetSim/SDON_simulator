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

    def __init__(self, properties: dict = None):
        self.engine_props = properties
        self.sdn_props = empty_props

        self.ai_obj = None
        self.snr_obj = SnrMeasurements(properties=properties)
        self.route_obj = Routing(engine_props=self.engine_props, sdn_props=self.sdn_props)
        self.spectrum_obj = SpectrumAssignment(engine_props=self.engine_props, sdn_props=self.sdn_props,
                                               route_props=self.route_obj.route_props)

    # TODO: Naming conventions here
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

            # TODO: It appears that is_free was never set to true yet it's true?
            if tmp_set != {0.0} or rev_tmp_set != {0.0}:
                raise BufferError("Attempted to allocate a taken spectrum.")

            core_matrix = link_dict['cores_matrix']
            rev_core_matrix = rev_link_dict['cores_matrix']
            core_matrix[core_num][start_slot:end_slot] = self.sdn_props['req_id']
            rev_core_matrix[core_num][start_slot:end_slot] = self.sdn_props['req_id']

            if self.engine_props['guard_slots']:
                self._allocate_gb(core_matrix=core_matrix, rev_core_matrix=rev_core_matrix, end_slot=end_slot,
                                  core_num=core_num)

    # TODO: Naming conventions here
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
                if self.route_obj.route_props['mod_formats_list'][path_index][0] is False:
                    self.sdn_props['was_routed'] = False
                    self.sdn_props['block_reason'] = 'distance'
                    return

                # TODO: Core was passed to spectrum because of the AI object, need to fix this
                # TODO: We have route props in spectrum, just use it there
                # TODO: Mod options are false?
                mod_options = self.route_obj.route_props['mod_formats_list'][path_index]
                # TODO: Need to keep track of XT cost
                self.spectrum_obj.spectrum_props['path_list'] = path_list
                self.spectrum_obj.get_spectrum(mod_options=mod_options)
                # Request was blocked for this path
                if self.spectrum_obj.spectrum_props['is_free'] is not True:
                    self.sdn_props['block_reason'] = 'congestion'
                    continue

                # TODO: Still not sure what to do here
                # TODO: Instead of doing this, just return routing props and or spectrum props into sdn props as a key
                #   Or, just return since we have that information in sdn_controller
                self.sdn_props['was_routed'] = True
                self.sdn_props['core_num'] = self.spectrum_obj.spectrum_props['core_num']
                self.sdn_props['path_list'] = path_list
                self.sdn_props['mod_format'] = self.spectrum_obj.spectrum_props['modulation']
                self.sdn_props['route_time'] = route_time
                self.sdn_props['path_weight'] = self.route_obj.route_props['weights_list'][path_index]
                self.sdn_props['is_sliced'] = False
                # TODO: Have a route and spectrum entry in sdn_props, easier that way
                self.sdn_props['start_slot'] = self.spectrum_obj.spectrum_props['start_slot']
                self.sdn_props['end_slot'] = self.spectrum_obj.spectrum_props['end_slot']
                # TODO: Always one until segment slicing is implemented
                self.sdn_props['num_trans'] = 1

                # TODO: This will have access to spectrum props
                self.allocate(self.spectrum_obj.spectrum_props['start_slot'],
                              self.spectrum_obj.spectrum_props['end_slot'],
                              self.spectrum_obj.spectrum_props['core_num'])
                return

        self.sdn_props['block_reason'] = 'distance'
        self.sdn_props['was_routed'] = False
