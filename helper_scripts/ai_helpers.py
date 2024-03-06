import numpy as np


# TODO: Remove AI object from all other sim scripts
# TODO: Check on reqs_dict


class AIHelpers:
    def __init__(self, ai_props: dict):
        self.ai_props = ai_props

        # TODO: Make sure to update these in run_dqn_sim
        self.topology = None
        self.net_spec_dict = None
        self.path_list = None
        self.core_num = None
        self.start_slot = None
        self.end_slot = None
        self.reqs_status_dict = None

    def update_net_spec_dict(self):
        for link_tuple in zip(self.path_list, self.path_list[1:]):
            rev_link_tuple = (link_tuple[1], link_tuple[0])

            req_id = self.ai_props['arrival_list'][self.ai_props['arrival_count']]['req_id']
            self.net_spec_dict[link_tuple]['cores_matrix'][self.core_num][self.start_slot:self.end_slot] = req_id
            self.net_spec_dict[rev_link_tuple]['cores_matrix'][self.core_num][self.start_slot:self.end_slot] = req_id

            self.net_spec_dict[link_tuple]['cores_matrix'][self.core_num][self.end_slot] = req_id * -1
            self.net_spec_dict[rev_link_tuple]['cores_matrix'][self.core_num][self.end_slot] = req_id * -1

    def check_is_free(self):
        is_free = True
        link_dict = None
        rev_link_dict = None
        for link_tuple in zip(self.path_list, self.path_list[1:]):
            rev_link_tuple = link_tuple[1], link_tuple[0]
            link_dict = self.net_spec_dict[link_tuple]
            rev_link_dict = self.net_spec_dict[rev_link_tuple]

            tmp_set = set(link_dict['cores_matrix'][self.core_num][self.start_slot:self.end_slot + 1])
            rev_tmp_set = set(rev_link_dict['cores_matrix'][self.core_num][self.start_slot:self.end_slot + 1])

            if tmp_set != {0.0} or rev_tmp_set != {0.0}:
                is_free = False

        return is_free, link_dict, rev_link_dict

    def _release(self, source_dest: tuple, dest_source: tuple, req_id_arr: np.array, gb_arr: np.array):
        for req_index, gb_index in zip(req_id_arr, gb_arr):
            self.net_spec_dict[source_dest]['cores_matrix'][self.core_num][req_index] = 0
            self.net_spec_dict[dest_source]['cores_matrix'][self.core_num][req_index] = 0

            self.net_spec_dict[source_dest]['cores_matrix'][self.core_num][gb_index] = 0
            self.net_spec_dict[dest_source]['cores_matrix'][self.core_num][gb_index] = 0

    def release(self, depart_time: float):
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
        curr_time = self.ai_props['arrival_list'][self.ai_props['arrival_count']]['arrive']
        index_list = list()

        for i, req_obj in enumerate(self.ai_props['depart_list']):
            if req_obj['depart'] <= curr_time:
                index_list.append(i)
                self.release(depart_time=req_obj['depart'])

        for index in index_list:
            self.ai_props['depart_list'].pop(index)

    def update_reqs_status(self, was_routed: bool, mod_format: str = None):
        self.reqs_status_dict.update(
            {self.ai_props['arrival_list'][self.ai_props['arrival_count']]['req_id']: {
                "mod_format": mod_format,
                "path": self.path_list,
                "is_sliced": False,
                "was_routed": was_routed,
            }})

    # TODO: Make this better after converting to a class (a loop?)
    def _allocate(self, is_free: bool, mod_format: str, bandwidth: str, path_len: float):
        if is_free:
            self.update_net_spec_dict()
            self.update_reqs_status(was_routed=True, mod_format=mod_format)

            self.ai_props['mock_sdn_dict']['bandwidth_list'].append(bandwidth)
            self.ai_props['mock_sdn_dict']['modulation_list'].append(mod_format)
            self.ai_props['mock_sdn_dict']['core_list'].append(self.core_num)
            self.ai_props['mock_sdn_dict']['path_weight'] = path_len
            self.ai_props['mock_sdn_dict']['spectrum_dict']['modulation'] = mod_format

            self.ai_props['mock_sdn_dict']['was_routed'] = True
            was_allocated = True
            return was_allocated

        was_allocated = False
        self.ai_props['mock_sdn_dict']['block_reason'] = 'congestion'
        self.ai_props['mock_sdn_dict']['was_routed'] = False
        return was_allocated

    # TODO: Break to more functions after this is a class
    def allocate(self, route_obj: object):
        was_allocated = True
        self.ai_props['mock_sdn_dict']['was_routed'] = True
        for path_index, path_list in enumerate(route_obj.route_props['paths_list']):
            self.path_list = path_list
            path_len = route_obj.route_props['weights_list'][path_index]
            mod_format = route_obj.route_props['mod_formats_list'][path_index][0]
            self.ai_props['mock_sdn_dict']['path_list'] = path_list
            if not mod_format:
                self.ai_props['mock_sdn_dict']['was_routed'] = False
                self.ai_props['mock_sdn_dict']['block_reason'] = 'distance'
                was_allocated = False
                continue

            bandwidth = self.ai_props['arrival_list'][self.ai_props['arrival_count']]['bandwidth']
            bandwidth_dict = self.ai_props['engine_props']['mod_per_bw'][bandwidth]
            self.end_slot = self.start_slot + bandwidth_dict[mod_format]['slots_needed']
            if self.end_slot >= self.ai_props['engine_props']['spectral_slots']:
                self.ai_props['mock_sdn_dict']['was_routed'] = False
                self.ai_props['mock_sdn_dict']['block_reason'] = 'congestion'
                was_allocated = False
                continue

            is_free, link_dict, rev_link_dict = self.check_is_free()
            was_allocated = self._allocate(is_free=is_free, mod_format=mod_format, bandwidth=bandwidth,
                                           path_len=path_len)

        self.update_reqs_status(was_routed=False)
        return was_allocated

    # TODO: Route time and number of transistors static
    def update_mock_sdn(self, mock_sdn: dict, curr_req: dict):
        self.ai_props['mock_sdn_dict'] = {
            'source': curr_req['source'],
            'destination': curr_req['destination'],
            'bandwidth': curr_req['bandwidth'],
            'net_spec_dict': self.net_spec_dict,
            'topology': self.topology,
            'mod_formats': curr_req['mod_formats'],
            # TODO: This number isn't correct in output?
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
