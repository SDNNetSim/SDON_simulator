import numpy as np


# TODO: Params to constructor or props: path_list, core_num, start_slot, end_slot, net_spec_dict?
# TODO: Objectify this to ai helpers object
# TODO: Remove AI object from all other sim scripts

def update_net_spec_dict(ai_props: dict, arrival_count: int, net_spec_dict: dict, path_list: list, core_num: int,
                         start_slot: int, end_slot: int):
    for link_tuple in zip(path_list, path_list[1:]):
        rev_link_tuple = (link_tuple[1], link_tuple[0])

        req_id = ai_props['arrival_list'][arrival_count]['req_id']
        net_spec_dict[link_tuple]['cores_matrix'][core_num][start_slot:end_slot] = req_id
        net_spec_dict[rev_link_tuple]['cores_matrix'][core_num][start_slot:end_slot] = req_id

        net_spec_dict[link_tuple]['cores_matrix'][core_num][end_slot] = req_id * -1
        net_spec_dict[rev_link_tuple]['cores_matrix'][core_num][end_slot] = req_id * -1


def check_is_free(net_spec_dict: dict, path_list: list, core_num: int, start_slot: int, end_slot: int):
    is_free = True
    link_dict = None
    rev_link_dict = None
    for link_tuple in zip(path_list, path_list[1:]):
        rev_link_tuple = link_tuple[1], link_tuple[0]
        link_dict = net_spec_dict[link_tuple]
        rev_link_dict = net_spec_dict[rev_link_tuple]

        tmp_set = set(link_dict['cores_matrix'][core_num][start_slot:end_slot + 1])
        rev_tmp_set = set(rev_link_dict['cores_matrix'][core_num][start_slot:end_slot + 1])

        if tmp_set != {0.0} or rev_tmp_set != {0.0}:
            is_free = False

    return is_free, link_dict, rev_link_dict


def _release(net_spec_dict: dict, source_dest: tuple, dest_source: tuple, core_num: int, req_id_arr: np.array,
             gb_arr: np.array):
    for req_index, gb_index in zip(req_id_arr, gb_arr):
        net_spec_dict[source_dest]['cores_matrix'][core_num][req_index] = 0
        net_spec_dict[dest_source]['cores_matrix'][core_num][req_index] = 0

        net_spec_dict[source_dest]['cores_matrix'][core_num][gb_index] = 0
        net_spec_dict[dest_source]['cores_matrix'][core_num][gb_index] = 0


def release(net_spec_dict: dict, engine_props: dict, reqs_dict: dict, reqs_status_dict: dict, depart_time: float):
    arrival_id = reqs_dict[depart_time]['req_id']
    if reqs_status_dict[arrival_id]['was_routed']:
        path_list = reqs_status_dict[arrival_id]['path']

        for source, dest in zip(path_list, path_list[1:]):
            source_dest = (source, dest)
            dest_source = (dest, source)

            for core_num in range(engine_props['cores_per_link']):
                core_arr = net_spec_dict[source_dest]['cores_matrix'][core_num]
                req_id_arr = np.where(core_arr == arrival_id)
                gb_arr = np.where(core_arr == (arrival_id * -1))

                _release(net_spec_dict=net_spec_dict, source_dest=source_dest, dest_source=dest_source,
                         core_num=core_num, req_id_arr=req_id_arr, gb_arr=gb_arr)
    # Request was blocked
    else:
        pass


def check_release(dqn_props: dict, arrival_count: int, net_spec_dict: dict, engine_props: dict, reqs_dict: dict,
                  reqs_status_dict: dict):
    curr_time = dqn_props['arrival_list'][arrival_count]['arrive']
    index_list = list()

    for i, req_obj in enumerate(dqn_props['depart_list']):
        if req_obj['depart'] <= curr_time:
            index_list.append(i)
            release(net_spec_dict=net_spec_dict, engine_props=engine_props, reqs_dict=reqs_dict,
                    reqs_status_dict=reqs_status_dict, depart_time=req_obj['depart'])

    for index in index_list:
        dqn_props['depart_list'].pop(index)


def update_reqs_status(reqs_status_dict: dict, dqn_props: dict, arrival_count: int, was_routed: bool,
                       mod_format: str = None, path_list: list = None):
    reqs_status_dict.update({dqn_props['arrival_list'][arrival_count]['req_id']: {
        "mod_format": mod_format,
        "path": path_list,
        "is_sliced": False,
        "was_routed": was_routed,
    }})


def _allocate(dqn_props: dict, net_spec_dict: dict, arrival_count: int, is_free: bool, path_list: list, start_slot: int,
              end_slot: int, core_num: int, mod_format: str, bandwidth: str, path_len: float, mock_sdn: dict,
              reqs_status_dict: dict):
    if is_free:
        update_net_spec_dict(ai_props=dqn_props, net_spec_dict=net_spec_dict,
                             arrival_count=arrival_count, path_list=path_list, core_num=core_num,
                             start_slot=start_slot, end_slot=end_slot)

        update_reqs_status(path_list=path_list, was_routed=True, mod_format=mod_format, arrival_count=arrival_count,
                           dqn_props=dqn_props, reqs_status_dict=reqs_status_dict)

        mock_sdn['bandwidth_list'].append(bandwidth)
        mock_sdn['modulation_list'].append(mod_format)
        mock_sdn['core_list'].append(core_num)
        mock_sdn['path_weight'] = path_len
        mock_sdn['spectrum_dict']['modulation'] = mod_format

        mock_sdn['was_routed'] = True
        was_allocated = True
        return was_allocated

    was_allocated = False
    mock_sdn['block_reason'] = 'congestion'
    mock_sdn['was_routed'] = False
    return was_allocated


def allocate(reqs_status_dict: dict, net_spec_dict: dict, arrival_count: int, engine_props: dict, dqn_props: dict,
             mock_sdn: dict, route_obj: object, core_num: int, start_slot: int):
    was_allocated = True
    mock_sdn['was_routed'] = True
    for path_index, path_list in enumerate(route_obj.route_props['paths_list']):
        path_len = route_obj.route_props['weights_list'][path_index]
        mod_format = route_obj.route_props['mod_formats_list'][path_index][0]
        mock_sdn['path_list'] = path_list
        if not mod_format:
            mock_sdn['was_routed'] = False
            mock_sdn['block_reason'] = 'distance'
            was_allocated = False
            continue

        bandwidth = dqn_props['arrival_list'][arrival_count]['bandwidth']
        bandwidth_dict = engine_props['mod_per_bw'][bandwidth]
        end_slot = start_slot + bandwidth_dict[mod_format]['slots_needed']
        if end_slot >= engine_props['spectral_slots']:
            mock_sdn['was_routed'] = False
            mock_sdn['block_reason'] = 'congestion'
            was_allocated = False
            continue

        is_free, link_dict, rev_link_dict = check_is_free(net_spec_dict=net_spec_dict,
                                                          path_list=path_list, core_num=core_num,
                                                          start_slot=start_slot, end_slot=end_slot)
        was_allocated = _allocate(is_free=is_free, path_list=path_list, start_slot=start_slot,
                                  end_slot=end_slot, core_num=core_num, mod_format=mod_format,
                                  bandwidth=bandwidth, path_len=path_len, arrival_count=arrival_count,
                                  dqn_props=dqn_props, mock_sdn=mock_sdn, net_spec_dict=net_spec_dict,
                                  reqs_status_dict=reqs_status_dict)

    update_reqs_status(dqn_props=dqn_props, arrival_count=arrival_count, was_routed=False,
                       reqs_status_dict=reqs_status_dict)
    return was_allocated


# TODO: Route time and number of transistors static
def update_mock_sdn(mock_sdn: dict, engine_obj: object, curr_req: dict):
    mock_sdn = {
        'source': curr_req['source'],
        'destination': curr_req['destination'],
        'bandwidth': curr_req['bandwidth'],
        'net_spec_dict': engine_obj.net_spec_dict,
        'topology': engine_obj.topology,
        'mod_formats': curr_req['mod_formats'],
        # TODO: This number isn't correct in output
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
