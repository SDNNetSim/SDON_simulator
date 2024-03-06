import numpy as np


class AIHelpers:
    """
    Contains methods to assist with AI simulations.
    """

    def __init__(self, ai_props: dict):
        self.ai_props = ai_props

        self.topology = None
        self.net_spec_dict = None
        self.reqs_status_dict = None

        self.path_list = None
        self.core_num = None
        self.start_slot = None
        self.end_slot = None
        self.mod_format = None
        self.path_len = None
        self.bandwidth = None

    def update_net_spec_dict(self):
        """
        Updates the network spectrum database.
        """
        for link_tuple in zip(self.path_list, self.path_list[1:]):
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
        for link_tuple in zip(self.path_list, self.path_list[1:]):
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
                "path": self.path_list,
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

    def _allocate(self, is_free: bool):
        if is_free:
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

    def _update_path_vars(self, route_obj: object, path_list: list, path_index: int):
        self.path_list = path_list
        self.path_len = route_obj.route_props['weights_list'][path_index]
        self.mod_format = route_obj.route_props['mod_formats_list'][path_index][0]
        self.ai_props['mock_sdn_dict']['path_list'] = path_list

    def allocate(self, route_obj: object):
        """
        Attempts to allocate a given request.

        :param route_obj: The Routing class.
        """
        was_allocated = True
        self.ai_props['mock_sdn_dict']['was_routed'] = True
        for path_index, path_list in enumerate(route_obj.route_props['paths_list']):
            self._update_path_vars(route_obj=route_obj, path_list=path_list, path_index=path_index)
            if not self.mod_format:
                self.ai_props['mock_sdn_dict']['was_routed'] = False
                self.ai_props['mock_sdn_dict']['block_reason'] = 'distance'
                was_allocated = False
                continue

            self._get_end_slot()
            if self.end_slot >= self.ai_props['engine_props']['spectral_slots']:
                self.ai_props['mock_sdn_dict']['was_routed'] = False
                self.ai_props['mock_sdn_dict']['block_reason'] = 'congestion'
                was_allocated = False
                continue

            is_free = self.check_is_free()
            was_allocated = self._allocate(is_free=is_free)

        self.update_reqs_status(was_routed=False)
        return was_allocated

    # TODO: Route time and number of transistors static
    #   - Check num transistors output
    def update_mock_sdn(self, curr_req: dict):
        """
        Updates the mock sdn dictionary.
        :param curr_req: The current request.
        """
        mock_sdn = {
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

        return mock_sdn
