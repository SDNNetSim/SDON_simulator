import time
import numpy as np

from arg_scripts.sdn_args import empty_props
from sim_scripts.snr_measurements import SnrMeasurements
from helper_scripts.routing_helpers import get_route
from helper_scripts.sim_helpers import get_spectrum


# TODO: Private methods don't really need comments
# TODO: Naming conventions, review all files with team docs
class SDNController:
    """
    This class contains methods to support software-defined network controller functionality.
    """

    def __init__(self, properties: dict = None):
        # TODO: Remember you changed sdn_props to engine_props
        self.engine_props = properties
        self.sdn_props = empty_props
        self.ai_obj = None
        self.snr_obj = SnrMeasurements(properties=properties)

    def release(self):
        """
        Removes a previously allocated request from the network.

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

    # TODO: Potentially split into smaller methods
    def allocate(self, start_slot: int, end_slot: int, core_num: int):
        """
        Allocates a network request.

        :param start_slot: The starting spectral slot to allocate the request
        :param end_slot: The ending spectral slot to allocate the request
        :param core_num: The desired core to allocate the request
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

    def handle_lps(self):
        raise NotImplementedError

    def _handle_spectrum(self, mod_options: dict, core: int):
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
                if self.sdn_props['max_segments'] > 1:
                    raise NotImplementedError

                continue

            spectrum, self.block_reason, xt_cost = get_spectrum(properties=self.sdn_props, chosen_bw=self.chosen_bw,
                                                                path=self.path,
                                                                net_spec_db=self.net_spec_db, modulation=modulation,
                                                                snr_obj=self.snr_obj,
                                                                path_mod=modulation, core=core)

            # We found a spectrum, no need to check other modulation formats
            if spectrum is not False:
                mod_chosen = modulation
                break

        return spectrum, xt_cost, mod_chosen

    def _handle_routing(self):
        resp = get_route(engine_props=self.engine_props, sdn_props=self.sdn_props, ai_obj=self.ai_obj)
        return resp

    def handle_event(self, request_type: str):
        """
        Handles any event that occurs in the simulation, controls this class.

        :param request_type: Whether the request is an arrival or departure.
        :return: The response with relevant information, network database, and physical topology
        """
        # Even if the request is blocked, we still use one transponder
        # TODO: Don't think we need to return this
        self.num_transponders = 1
        self.block_reason = None

        if request_type == "release":
            self.release()
            return self.net_spec_db

        start_time = time.time()
        paths, cores, path_mods, path_weights = self._handle_routing()
        route_time = time.time() - start_time

        for path, core, path_mod, path_weight in zip(paths, cores, path_mods, path_weights):
            self.path = path
            self.core = core

            # TODO: Spectrum assignment always overrides modulation format chosen when using check snr
            # TODO: Fix this up
            if path is not False:
                if self.sdn_props['check_snr'] != 'None' and self.sdn_props['check_snr'] is not None:
                    raise ValueError('You must check that max lengths are not zero before running this.')
                if path_mod is not False:
                    # TODO: Fix this bug
                    mod_options = path_mod
                else:
                    self.block_reason = 'distance'
                    return False, self.block_reason, self.path

                spectrum, xt_cost, modulation = self._handle_spectrum(mod_options=mod_options, core=core)
                # Request was blocked for this path
                if spectrum is False or spectrum is None:
                    self.block_reason = 'congestion'
                    continue

                resp = {
                    'path': self.path,
                    'mod_format': modulation,
                    'route_time': route_time,
                    'path_weight': path_weight,
                    'xt_cost': xt_cost,
                    'is_sliced': False,
                    'spectrum': spectrum
                }
                self.allocate(spectrum['start_slot'], spectrum['end_slot'], spectrum['core_num'])
                return resp, self.net_spec_db, self.num_transponders, self.path

            self.block_reason = 'distance'

        return False, self.block_reason, self.path
