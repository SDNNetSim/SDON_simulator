import copy

import numpy as np

from sim_scripts.spectrum_assignment import SpectrumAssignment
from sim_scripts.sdn_controller import SDNController
from helper_scripts.sim_helpers import find_path_len


class AIHelpers:
    """
    Contains methods to assist with AI simulations.
    """

    # TODO: Update all constructor params (ensure they are used)
    def __init__(self, ai_props: dict, engine_obj: object):
        self.ai_props = ai_props
        # TODO: Use this
        self.engine_obj = engine_obj

        self.topology = None
        self.net_spec_dict = None
        self.reqs_status_dict = None

        self.mock_sdn = None
        self.path_matrix = None
        self.path_index = None
        self.core_num = None
        self.start_slot = None
        self.end_slot = None
        self.slice_request = None
        self.mod_format = None
        self.path_len = None
        self.bandwidth = None

        self.best_fit_params = {'start_slot': None, 'core_num': None}

        # TODO: Use this, note that net spec db will probably be in here
        self.sdn_obj = SDNController(engine_props=self.ai_props['engine_props'])

    def update_net_spec_dict(self):
        """
        Updates the network spectrum database.
        """
        if self.ai_props['engine_props']['guard_slots']:
            self.end_slot = self.end_slot - 1
        else:
            raise NotImplementedError

        # TODO: here we don't actually check anything....Use sdn controller and nothing else
        for link_tuple in zip(self.path_matrix, self.path_matrix[1:]):
            rev_link_tuple = (link_tuple[1], link_tuple[0])

            req_id = self.ai_props['arrival_list'][self.ai_props['arrival_count']]['req_id']
            self.net_spec_dict[link_tuple]['cores_matrix'][self.core_num][self.start_slot:self.end_slot] = req_id
            self.net_spec_dict[rev_link_tuple]['cores_matrix'][self.core_num][self.start_slot:self.end_slot] = req_id

            self.net_spec_dict[link_tuple]['cores_matrix'][self.core_num][self.end_slot] = req_id * -1
            self.net_spec_dict[rev_link_tuple]['cores_matrix'][self.core_num][self.end_slot] = req_id * -1

    def check_is_free(self):
        """
        Checks if a spectrum is available along the given path.
        """
        is_free = True
        for link_tuple in zip(self.path_matrix, self.path_matrix[1:]):
            rev_link_tuple = link_tuple[1], link_tuple[0]
            link_dict = self.net_spec_dict[link_tuple]
            rev_link_dict = self.net_spec_dict[rev_link_tuple]

            tmp_set = set(link_dict['cores_matrix'][self.core_num][self.start_slot:self.end_slot + 1])
            rev_tmp_set = set(rev_link_dict['cores_matrix'][self.core_num][self.start_slot:self.end_slot + 1])

            if tmp_set != {0.0} or rev_tmp_set != {0.0}:
                is_free = False

        return is_free

    def _release(self, source_dest: tuple, dest_source: tuple, req_id_arr: np.array, gb_arr: np.array):
        for req_index, gb_index in zip(req_id_arr, gb_arr):
            self.net_spec_dict[source_dest]['cores_matrix'][self.core_num][req_index] = 0
            self.net_spec_dict[dest_source]['cores_matrix'][self.core_num][req_index] = 0

            self.net_spec_dict[source_dest]['cores_matrix'][self.core_num][gb_index] = 0
            self.net_spec_dict[dest_source]['cores_matrix'][self.core_num][gb_index] = 0

    # TODO: Call release in engine instead, should have the correct net spec dict!
    #   - Ensure net spec dict is updted properly
    def release(self, depart_time: float):
        """
        Releases a given request.
        :param depart_time: The departure time of the request.
        """
        arrival_id = self.ai_props['reqs_dict'][depart_time]['req_id']
        if self.reqs_status_dict[arrival_id]['was_routed']:
            path_list = self.reqs_status_dict[arrival_id]['path']

            for source, dest in zip(path_list, path_list[1:]):
                source_dest = (source, dest)
                dest_source = (dest, source)

                for core_num in range(self.ai_props['engine_props']['cores_per_link']):
                    core_arr = self.net_spec_dict[source_dest]['cores_matrix'][core_num]
                    req_id_arr = np.where(core_arr == arrival_id)
                    gb_arr = np.where(core_arr == (arrival_id * -1))

                    self._release(source_dest=source_dest, dest_source=dest_source, req_id_arr=req_id_arr,
                                  gb_arr=gb_arr)
        # Request was blocked
        else:
            pass

    def check_release(self):
        """
        Checks if a request or multiple requests need to be released.
        """
        curr_time = self.ai_props['arrival_list'][self.ai_props['arrival_count']]['arrive']
        index_list = list()

        for i, req_obj in enumerate(self.ai_props['depart_list']):
            if req_obj['depart'] <= curr_time:
                index_list.append(i)
                # TODO:
                self.release(depart_time=req_obj['depart'])

        for index in index_list:
            self.ai_props['depart_list'].pop(index)

    def update_reqs_status(self, was_routed: bool):
        """
        Updates a given request's status.
        :param was_routed: If the request was successfully routed or not.
        """
        self.reqs_status_dict.update(
            {self.ai_props['arrival_list'][self.ai_props['arrival_count']]['req_id']: {
                "mod_format": self.mod_format,
                "path": self.path_matrix,
                # TODO: This is not true
                "is_sliced": False,
                "was_routed": was_routed,
            }})

    def _allocate_mock_sdn(self):
        self.ai_props['mock_sdn_dict']['bandwidth_list'].append(self.bandwidth)
        self.ai_props['mock_sdn_dict']['modulation_list'].append(self.mod_format)
        self.ai_props['mock_sdn_dict']['core_list'].append(self.core_num)
        self.ai_props['mock_sdn_dict']['path_weight'] = self.path_len
        self.ai_props['mock_sdn_dict']['spectrum_dict']['modulation'] = self.mod_format
        self.ai_props['mock_sdn_dict']['was_routed'] = True

    def _get_best_fit(self):
        mock_engine_props = copy.deepcopy(self.ai_props['engine_props'])
        mock_engine_props['allocation_method'] = 'best_fit'
        mock_engine_props['ai_algorithm'] = 'None'
        mock_sdn_props = {
            'net_spec_dict': self.net_spec_dict,
            'mod_formats': self.ai_props['engine_props']['mod_per_bw'][self.bandwidth]
        }
        spectrum_obj = SpectrumAssignment(engine_props=mock_engine_props, sdn_props=mock_sdn_props)
        # TODO: Has forced core
        spectrum_obj.spectrum_props['forced_core'] = self.core_num

        spectrum_obj.spectrum_props['path_list'] = self.path_matrix
        mod_format_list = [self.mod_format]
        spectrum_obj.get_spectrum(mod_format_list=mod_format_list, ai_obj=None)

        for spectrum_param in ('core_num', 'start_slot', 'end_slot'):
            self.best_fit_params[spectrum_param] = spectrum_obj.spectrum_props[spectrum_param]

    def _allocate(self, is_free: bool):
        if is_free:
            self._get_best_fit()
            self.update_net_spec_dict()
            self.update_reqs_status(was_routed=True)
            self._allocate_mock_sdn()

            was_allocated = True
            return was_allocated

        was_allocated = False
        self.ai_props['mock_sdn_dict']['block_reason'] = 'congestion'
        self.ai_props['mock_sdn_dict']['was_routed'] = False
        return was_allocated

    def _get_end_slot(self):
        self.bandwidth = self.ai_props['arrival_list'][self.ai_props['arrival_count']]['bandwidth']
        bandwidth_dict = self.ai_props['engine_props']['mod_per_bw'][self.bandwidth]
        self.end_slot = self.start_slot + bandwidth_dict[self.mod_format]['slots_needed']

    def _handle_slicing(self):
        # fixme Taking an allocated spectrum, I think there is a discrepancy between net spec dbs
        sdn_obj = SDNController(engine_props=self.ai_props['engine_props'])
        sdn_obj._init_req_stats()
        # TODO: Here is where net spec db is
        sdn_obj.sdn_props = self.mock_sdn
        sdn_obj._handle_slicing(path_list=self.path_matrix)

        if sdn_obj.sdn_props['was_routed']:
            path_weight = find_path_len(path_list=sdn_obj.sdn_props['path_list'],
                                        topology=self.ai_props['engine_props']['topology'])
            self.ai_props['mock_sdn_dict']['path_weight'] = path_weight
            was_allocated = True
            return was_allocated

        was_allocated = False
        self.ai_props['mock_sdn_dict']['block_reason'] = 'congestion'
        self.ai_props['mock_sdn_dict']['was_routed'] = False
        return was_allocated

    def _update_path_vars(self, route_obj: object, path_matrix: list):
        self.path_matrix = path_matrix
        self.path_len = route_obj.route_props['weights_list'][self.path_index]
        self.mod_format = route_obj.route_props['mod_formats_list'][self.path_index][0]
        self.ai_props['mock_sdn_dict']['path_list'] = path_matrix

    def allocate(self, route_obj: object):
        """
        Attempts to allocate a given request.

        :param route_obj: The Routing class.
        """
        # TODO: Ensure path index is correct
        # TODO: Ensure network spectrum is update properly
        # TODO: Ensure request ID is correct
        # TODO: Do not forget to use the SDN Controller to release, move release code here
        # TODO: Need to update with respect to engine handle arrival
        # TODO: Update arrival params like engine
        path_matrix = [route_obj.route_props['paths_list'][self.path_index]]
        # TODO: Not sure if I need this anymore
        self._update_path_vars(route_obj=route_obj, path_matrix=path_matrix)

        sdn_obj = SDNController(engine_props=self.ai_props['engine_props'])
        # TODO: Need access to dqn props
        for req_key, req_value in self.ai_props['reqs_dict'][curr_time].items():
            self.sdn_obj.sdn_props[req_key] = req_value
        sdn_obj.handle_event(request_type='arrival', force_slicing=self.slice_request, force_route_matrix=path_matrix)
        self.net_spec_dict = sdn_obj.sdn_props['net_spec_dict']


        for path_index, path_list in enumerate(route_obj.route_props['paths_list']):
            # Only consider the path selected by the agent
            if path_index != self.path_index:
                continue
            self._update_path_vars(route_obj=route_obj, path_list=path_list, path_index=path_index)
            if self.ai_props['engine_props']['max_segments'] > 1 and self.slice_request:
                was_allocated = self._handle_slicing()
                self.update_reqs_status(was_routed=was_allocated)
                return was_allocated
            elif not self.mod_format:
                self.ai_props['mock_sdn_dict']['was_routed'] = False
                self.ai_props['mock_sdn_dict']['block_reason'] = 'distance'
                was_allocated = False
                continue
            else:
                self.bandwidth = self.mock_sdn['bandwidth']
                self._get_best_fit()
                self.start_slot = self.best_fit_params['start_slot']
                self.end_slot = self.best_fit_params['end_slot']
                self.update_net_spec_dict()
                self._allocate_mock_sdn()

        # TODO: Changed this, always used to be false
        # TODO: Probably don't need this and many other things due to using net spec dict
        self.update_reqs_status(was_routed=was_allocated)
        return was_allocated

    # TODO: Route time and number of transistors static
    #   - Check num transistors output
    def update_mock_sdn(self, curr_req: dict):
        """
        Updates the mock sdn dictionary.
        :param curr_req: The current request.
        """
        self.mock_sdn = {
            'req_id': curr_req['req_id'],
            'source': curr_req['source'],
            'destination': curr_req['destination'],
            'bandwidth': curr_req['bandwidth'],
            'net_spec_dict': self.net_spec_dict,
            'topology': self.topology,
            'mod_formats': curr_req['mod_formats'],
            'num_trans': 1.0,
            'route_time': 0.0,
            'block_reason': None,
            'stat_key_list': ['modulation_list', 'xt_list', 'core_list'],
            'modulation_list': list(),
            'xt_list': list(),
            'is_sliced': False,
            'core_list': list(),
            'bandwidth_list': list(),
            'path_weight': list(),
            'spectrum_dict': {'modulation': None}
        }

        return self.mock_sdn
