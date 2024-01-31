import os
import json


class PlotHelpers:
    def __init__(self, plot_props: dict):
        self.plot_props = plot_props

    @staticmethod
    def list_to_title(input_list: list):
        if not input_list:
            return ""

        unique_list = list()
        for item in input_list:
            if item[0] not in unique_list:
                unique_list.append(item[0])

        if len(unique_list) > 1:
            return ", ".join(unique_list[:-1]) + " & " + unique_list[-1]

        return unique_list[0]

    def get_file_info(self):
        resp = dict()
        for lst_idx, (networks, dates, times) in enumerate(zip(self.net_names, self.dates, self.times)):
            for sim_index, curr_time in enumerate(times):
                resp[curr_time] = {'network': networks[sim_index], 'date': dates[sim_index], 'sims': dict()}
                curr_dir = f'{self.base_dir}/{networks[sim_index]}/{dates[sim_index]}/{curr_time}'

                # Sort by sim number
                sim_dirs = os.listdir(curr_dir)
                sim_dirs = sorted(sim_dirs, key=lambda x: int(x[1:]))

                for curr_sim in sim_dirs:
                    # User selected to not run this simulation
                    if curr_sim not in self.sims[lst_idx]:
                        continue
                    curr_fp = os.path.join(curr_dir, curr_sim)
                    resp[curr_time]['sims'][curr_sim] = list()
                    files = os.listdir(curr_fp)
                    sorted_files = sorted(files, key=lambda x: float(x.split('_')[0]))

                    for erlang_file in sorted_files:
                        resp[curr_time]['sims'][curr_sim].append(erlang_file.split('_')[0])

        return resp


@staticmethod
def _snake_to_title(snake_str: str):
    words = snake_str.split('_')
    title_str = ' '.join(word.capitalize() for word in words)
    return title_str


@staticmethod
def _int_to_string(number):
    return '{:,}'.format(number)  # pylint: disable=consider-using-f-string


@staticmethod
def _dict_to_list(data_dict, nested_key, path=None, find_mean=False):
    if path is None:
        path = []

    extracted_values = []
    for value in data_dict.values():
        for key in path:
            value = value.get(key, {})
        nested_value = value.get(nested_key)
        if nested_value is not None:
            extracted_values.append(nested_value)

    if find_mean:
        return np.mean(extracted_values)

    return np.array(extracted_values)


def _find_sim_info(self, network):
    self.plot_dict[self.time][self.sim_num]['holding_time'] = self.erlang_dict['sim_params']['holding_time']
    self.plot_dict[self.time][self.sim_num]['cores_per_link'] = self.erlang_dict['sim_params']['cores_per_link']
    self.plot_dict[self.time][self.sim_num]['spectral_slots'] = self.erlang_dict['sim_params']['spectral_slots']
    self.plot_dict[self.time][self.sim_num]['network'] = network

    self.num_requests = self._int_to_string(self.erlang_dict['sim_params']['num_requests'])
    self.num_cores = self.erlang_dict['sim_params']['cores_per_link']


def _find_modulation_info(self):
    mod_usage = self.erlang_dict['misc_stats']['0']['modulation_formats']
    for bandwidth, mod_obj in mod_usage.items():
        for modulation in mod_obj.keys():
            filters = ['modulation_formats', bandwidth]
            mod_usages = self._dict_to_list(self.erlang_dict['misc_stats'], modulation, filters)

            modulation_info = self.plot_dict[self.time][self.sim_num]['modulations']
            if bandwidth not in modulation_info.keys():
                modulation_info[bandwidth] = dict()
                if modulation not in modulation_info[bandwidth].keys():
                    modulation_info[bandwidth][modulation] = [np.mean(mod_usages)]
                else:
                    modulation_info[bandwidth][modulation].append(np.mean(mod_usages))
            else:
                if modulation not in modulation_info[bandwidth].keys():
                    modulation_info[bandwidth][modulation] = [np.mean(mod_usages)]
                else:
                    modulation_info[bandwidth][modulation].append(np.mean(mod_usages))


def _find_misc_stats(self):
    path_lens = self._dict_to_list(self.erlang_dict['misc_stats'], 'mean', ['path_lengths'])
    hops = self._dict_to_list(self.erlang_dict['misc_stats'], 'mean', ['hops'])
    times = self._dict_to_list(self.erlang_dict['misc_stats'], 'route_times') * 10 ** 3

    cong_block = self._dict_to_list(self.erlang_dict['misc_stats'], 'congestion', ['block_reasons'])
    dist_block = self._dict_to_list(self.erlang_dict['misc_stats'], 'distance', ['block_reasons'])

    average_len = np.mean(path_lens)
    average_hop = np.mean(hops)
    average_times = np.mean(times)
    average_cong_block = np.mean(cong_block)
    average_dist_block = np.mean(dist_block)

    self.plot_dict[self.time][self.sim_num]['path_lengths'].append(average_len)
    self.plot_dict[self.time][self.sim_num]['hops'].append(average_hop)
    self.plot_dict[self.time][self.sim_num]['route_times'].append(average_times)
    self.plot_dict[self.time][self.sim_num]['cong_block'].append(average_cong_block)
    self.plot_dict[self.time][self.sim_num]['dist_block'].append(average_dist_block)


def _find_blocking(self):
    try:
        blocking_mean = self.erlang_dict['blocking_mean']
        if blocking_mean is None:
            blocking_mean = np.mean(list(self.erlang_dict['block_per_sim'].values()))
    except KeyError:
        blocking_mean = np.mean(list(self.erlang_dict['block_per_sim'].values()))
    self.plot_dict[self.time][self.sim_num]['blocking_vals'].append(blocking_mean)


def _find_network_usage(self):
    request_nums = []
    active_reqs = []
    block_per_req = []
    occ_slots = []
    for iteration in self.erlang_dict['misc_stats']:
        curr_obj = self.erlang_dict['misc_stats'][iteration]['request_snapshots']
        request_nums = [int(req_num) for req_num in list(curr_obj.keys())]
        curr_actives = self._dict_to_list(curr_obj, 'active_requests')
        curr_blocks = self._dict_to_list(curr_obj, 'blocking_prob')
        curr_slots = self._dict_to_list(curr_obj, 'occ_slots')

        active_reqs.append(np.nan_to_num(curr_actives))
        block_per_req.append(np.nan_to_num(curr_blocks))
        occ_slots.append(np.nan_to_num(curr_slots))

    self.plot_dict[self.time][self.sim_num]['req_nums'] = request_nums
    self.plot_dict[self.time][self.sim_num]['active_reqs'].append(np.mean(active_reqs, axis=0))
    self.plot_dict[self.time][self.sim_num]['block_per_req'].append(np.mean(block_per_req, axis=0))
    self.plot_dict[self.time][self.sim_num]['network_util'].append(np.mean(occ_slots, axis=0))


def _not_filters(filter_dict: dict, file_dict: dict):
    keep_config = True
    for flags_list in filter_dict['not_filter_list']:
        params_dict = file_dict.get('sim_params')
        keys_list = flags_list[0:-1]
        check_value = flags_list[-1]

        for curr_key in keys_list:
            params_dict = params_dict.get(curr_key)

        if params_dict == check_value:
            keep_config = False
            break

        keep_config = True

    return keep_config


def _or_filters(filter_dict: dict, file_dict: dict):
    keep_config = True
    for flags_list in filter_dict['or_filter_list']:
        params_dict = file_dict.get('sim_params')
        keys_list = flags_list[0:-1]
        check_value = flags_list[-1]

        for curr_key in keys_list:
            params_dict = params_dict.get(curr_key)

        if params_dict == check_value:
            keep_config = True
            break

        keep_config = False

    return keep_config


def _and_filters(filter_dict: dict, file_dict: dict):
    keep_config = True
    params_dict = file_dict.get('sim_params')
    for flags_list in filter_dict['and_filter_list']:
        keys_list = flags_list[0:-1]
        check_value = flags_list[-1]

        for curr_key in keys_list:
            params_dict = params_dict.get(curr_key)

        if params_dict != check_value:
            keep_config = False
            break

    return keep_config


def _check_filters(file_dict: dict, filter_dict: dict):
    keep_config = _and_filters(filter_dict=filter_dict, file_dict=file_dict)

    if keep_config:
        keep_config = _or_filters(filter_dict=filter_dict, file_dict=file_dict)

        if keep_config:
            keep_config = _not_filters(filter_dict=filter_dict, file_dict=file_dict)

    return keep_config


def find_times(dates_dict: dict, filter_dict: dict):
    resp = {
        'times_list': list(),
        'sims_list': list(),
        'networks_list': list(),
        'dates_list': list(),
    }
    info_dict = dict()
    for date, network in dates_dict.items():
        times_path = os.path.join('..', 'data', 'output', network, date)
        times_list = [curr_dir for curr_dir in os.listdir(times_path)
                      if os.path.isdir(os.path.join(times_path, curr_dir))]

        for curr_time in times_list:
            sims_path = os.path.join(times_path, curr_time)
            sim_num_list = [curr_dir for curr_dir in os.listdir(sims_path) if
                            os.path.isdir(os.path.join(sims_path, curr_dir))]

            for sim in sim_num_list:
                file_path = os.path.join(sims_path, sim)
                files = [file for file in os.listdir(file_path)
                         if os.path.isfile(os.path.join(file_path, file))]
                # All Erlangs will have the same configuration, just take the first file's config
                file_name = os.path.join(file_path, files[0])
                with open(file_name, 'r', encoding='utf-8') as file_obj:
                    file_dict = json.load(file_obj)

                keep_config = _check_filters(file_dict=file_dict, filter_dict=filter_dict)
                if keep_config:
                    if curr_time not in dates_dict:
                        info_dict[curr_time] = {'sim_list': list(), 'network_list': list(), 'dates_list': list()}
                    info_dict[curr_time]['sim_list'].append(sim)
                    info_dict[curr_time]['network_list'].append(network)
                    info_dict[curr_time]['dates_list'].append(date)

    # Convert info dict to lists
    for time, obj in info_dict.items():
        resp['times_list'].append([time])
        resp['sims_list'].append(obj['sim_list'])
        resp['networks_list'].append(obj['network_list'])
        resp['dates_list'].append(obj['dates_list'])

    return resp
