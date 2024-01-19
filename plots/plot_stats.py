# Standard library imports
import os
import json
import re

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np

# Local application imports
from helper_scripts.os_helpers import create_dir
from interactive_plots import plot_q_table


class PlotStats:
    """
    A class for computing and plotting statistical analysis for simulations.
    """

    def __init__(self, net_names: list, dates: list, times: list, sims: list):
        """
        Initializes the PlotStats class.
        """
        # Information to retrieve desired data
        self.net_names = net_names
        self.title_names = self._list_to_title(input_list=self.net_names)
        self.dates = dates
        self.times = times
        self.sims = sims
        self.base_dir = '../data/output'
        self.file_info = self._get_file_info()

        # The final dictionary containing information for all plots
        self.plot_dict = None
        self.time = None
        self.sim_num = None
        self.erlang_dict = None
        self.num_requests = None
        self.num_cores = None
        self.erlang = None
        self.network = None
        self.date = None
        # Miscellaneous customizations visually
        self.colors = ['#024de3', '#00b300', 'orange', '#6804cc', '#e30220']
        self.line_styles = ['solid', 'dashed', 'dotted', 'dashdot']
        self.markers = ['o', '^', 's', 'x']
        # self.x_ticks = list(range(10, 450, 50))
        self.x_ticks = [10, 50, 100, 150, 200, 250, 300, 350, 400]

        self._get_data()

    def _get_file_info(self):
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
    def _list_to_title(input_list: list):
        if not input_list:  # handle the case for an empty list
            return ""

        # Remove duplicates but maintain order
        unique_list = []
        for item in input_list:
            if item[0] not in unique_list:
                unique_list.append(item[0])

        # If after removing duplicates there's more than one string, use "&" before the last string
        if len(unique_list) > 1:
            return ", ".join(unique_list[:-1]) + " & " + unique_list[-1]

        # If only one string in the list (or all were duplicates of the same string)
        return unique_list[0]

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

    @staticmethod
    def _policy_to_short_form(policy_str):
        word_to_number = {
            'one': '1',
            'two': '2',
            'three': '3',
            'four': '4',
            'five': '5',
            'six': '6',
            'seven': '7',
            'eight': '8',
            'nine': '9',
        }
        last_word = policy_str.split('_')[-1]
        number = word_to_number.get(last_word)
        if number is None:
            return 'Baseline'

        return f"P{number}"

    def _find_algorithm(self):
        # num_requests = str(self.erlang_dict['sim_params']['num_requests'])[:2]
        route_method = self._snake_to_title(snake_str=self.erlang_dict['sim_params']['route_method'])
        # alloc_method = self._snake_to_title(snake_str=self.erlang_dict['sim_params']['allocation_method'])

        if route_method == 'Ai':
            # policy = self._snake_to_title(snake_str=self.erlang_dict['sim_params']['ai_arguments']['policy'])
            # param = f"Policy={policy}"
            learn_rate = self.erlang_dict['sim_params']['ai_arguments']['learn_rate']
            discount = self.erlang_dict['sim_params']['ai_arguments']['discount']
            policy = self.erlang_dict['sim_params']['ai_arguments']['policy']
            policy = self._policy_to_short_form(policy_str=policy)
            param = rf"{policy} $\alpha$={learn_rate} | $\gamma$={discount}"
        elif route_method == 'Xt Aware':
            xt_type = self.erlang_dict['sim_params']['xt_type']
            if xt_type == 'with_length':
                param = f"$\\beta$={self.erlang_dict['sim_params']['beta']}"
            else:
                param = "Span * XT"
        else:
            param = f"k={self.erlang_dict['sim_params']['k_paths']}"

        algorithm = f'{param}'
        self.plot_dict[self.time][self.sim_num]['algorithm'] = algorithm

    def _get_tables(self):
        base_path = f"../ai/models/q_tables/{self.network}/{self.date}"
        des_path = os.path.join(base_path, f"{self.time}/{self.sim_num}/"
                                           f"{self.erlang}_routes_table_c{self.num_cores}.npy")
        q_table = np.load(des_path, allow_pickle=True)
        return q_table

    def _get_rewards(self):
        base_path = f"../ai/models/q_tables/{self.network}/{self.date}"
        des_path = os.path.join(base_path, f"{self.time}/{self.sim_num}/{self.erlang}_params_c{self.num_cores}.json")

        with open(des_path, 'r', encoding='utf-8') as file_obj:
            file_content = json.load(file_obj)

        if len(self.plot_dict[self.time][self.sim_num]['epsilon']) == 0:
            self.plot_dict[self.time][self.sim_num]['epsilon'] = file_content['epsilon_decay'][0:-1]

        reward_info = file_content['reward_info']
        min_rewards = (reward_info['routes']['min'], reward_info['cores']['min'])
        max_rewards = (reward_info['routes']['max'], reward_info['cores']['max'])
        average_rewards = (reward_info['routes']['average'], reward_info['cores']['average'])

        error_info = file_content['td_info']
        min_error = (error_info['routes']['min'], error_info['cores']['min'])
        max_error = (error_info['routes']['max'], error_info['cores']['max'])
        average_error = (error_info['routes']['average'], error_info['cores']['average'])

        cumulative_sum = 0
        cumulative_rewards = []
        cumulative_errors = []

        for i, reward in enumerate(average_rewards[0], 1):
            cumulative_sum += reward
            cumulative_rewards.append(cumulative_sum / i)

        cumulative_sum = 0
        for i, reward in enumerate(average_error[0], 1):
            cumulative_sum += reward
            cumulative_errors.append(cumulative_sum / i)

        return min_rewards, max_rewards, cumulative_rewards, min_error, max_error, cumulative_errors

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

    def _find_ai_stats(self):
        min_rewards, max_rewards, average_rewards, min_error, max_error, average_error = self._get_rewards()
        self.plot_dict[self.time][self.sim_num]['min_rewards']['routes'].append(min_rewards[0])
        self.plot_dict[self.time][self.sim_num]['min_rewards']['cores'].append(min_rewards[1])

        self.plot_dict[self.time][self.sim_num]['max_rewards']['routes'].append(max_rewards[0])
        self.plot_dict[self.time][self.sim_num]['max_rewards']['cores'].append(max_rewards[1])

        self.plot_dict[self.time][self.sim_num]['average_rewards']['routes'].append(average_rewards)
        self.plot_dict[self.time][self.sim_num]['average_rewards']['cores'].append(average_rewards)

        self.plot_dict[self.time][self.sim_num]['min_error']['routes'].append(min_error[0])
        self.plot_dict[self.time][self.sim_num]['min_error']['cores'].append(min_error[1])

        self.plot_dict[self.time][self.sim_num]['max_error']['routes'].append(max_error[0])
        self.plot_dict[self.time][self.sim_num]['max_error']['cores'].append(max_error[1])

        self.plot_dict[self.time][self.sim_num]['average_error']['routes'].append(average_error)
        self.plot_dict[self.time][self.sim_num]['average_error']['cores'].append(average_error)

        self.plot_dict[self.time][self.sim_num]['time_steps'].append(np.arange(len(average_rewards)))

        q_table = self._get_tables()
        self.plot_dict[self.time][self.sim_num]['q_tables'].append(q_table)

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

    def _update_plot_dict(self):
        if self.plot_dict is None:
            self.plot_dict = {self.time: {}}
        elif self.time not in self.plot_dict:
            self.plot_dict[self.time] = {}

        self.plot_dict[self.time][self.sim_num] = {
            'erlang_vals': [],
            'blocking_vals': [],
            'path_lengths': [],
            'hops': [],
            'network_util': [],
            'active_reqs': [],
            'block_per_req': [],
            'req_nums': [],
            'route_times': [],
            'epsilon': [],
            'q_tables': [],
            'min_rewards': {'routes': [], 'cores': []},
            'max_rewards': {'routes': [], 'cores': []},
            'modulations': dict(),
            'average_rewards': {'routes': [], 'cores': []},
            'max_error': {'routes': [], 'cores': []},
            'min_error': {'routes': [], 'cores': []},
            'average_error': {'routes': [], 'cores': []},
            'time_steps': [],
            'dist_block': [],
            'cong_block': [],
            'holding_time': None,
            'cores_per_link': None,
            'spectral_slots': None,
        }

    def _get_data(self):
        for time, obj in self.file_info.items():
            self.time = time
            for curr_sim, erlang_vals in obj['sims'].items():
                self.sim_num = curr_sim
                self._update_plot_dict()
                for erlang in erlang_vals:
                    self.erlang = erlang
                    self.network = obj['network']
                    self.date = obj['date']
                    curr_fp = os.path.join(self.base_dir, obj['network'], obj['date'], time, curr_sim,
                                           f'{erlang}_erlang.json')
                    with open(curr_fp, 'r', encoding='utf-8') as file_obj:
                        erlang_dict = json.load(file_obj)

                    erlang = int(erlang.split('.')[0])
                    self.plot_dict[self.time][self.sim_num]['erlang_vals'].append(erlang)

                    self.erlang_dict = erlang_dict
                    self._find_blocking()
                    # self._find_algorithm()
                    # self._find_sim_info(network=obj['network'])
                    # self._find_modulation_info()
                    # self._find_network_usage()
                    # if self.erlang_dict['sim_params']['ai_algorithm'] != 'None':
                    #     self._find_ai_stats()
                    # self._find_misc_stats()

    def _save_plot(self, file_name: str):
        file_path = f'./output/{self.net_names[0][-1]}/{self.dates[0][-1]}/{self.times[0][-1]}'
        create_dir(file_path)
        plt.savefig(f'{file_path}/{file_name}_core{self.num_cores}.png')

    def _setup_plot(self, title, y_label, x_label, grid=True, y_ticks=True, y_lim=False, x_ticks=True):
        plt.figure(figsize=(7, 5), dpi=300)
        plt.title(f"{self.title_names} {title} R={self.num_requests}")
        plt.ylabel(y_label)
        plt.xlabel(x_label)

        if y_ticks:
            plt.yticks([10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1])
            plt.ylim(10 ** -4, 1)
            plt.yscale('log')

        if y_lim:
            plt.ylim(y_lim[0], y_lim[1])

        if x_ticks:
            plt.xticks(self.x_ticks)
            plt.xlim(self.x_ticks[0], self.x_ticks[-1])

        if grid:
            plt.grid()

    def _plot_sub_plots(self, plot_info, title, x_ticks, x_lim, x_label, y_ticks, y_lim, y_label, legend, filename):
        _, axes = plt.subplots(2, 2, figsize=(7, 5), dpi=300)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

        index_list = [[0, 0], [0, 1], [1, 0], [1, 1]]

        for idx, (_, obj) in enumerate(self.plot_dict.items()):
            for (x_key, y_key, sub_key, colors) in plot_info:
                curr_ax = axes[index_list[idx][0], index_list[idx][1]]
                curr_ax.set_yticks(y_ticks)

                if sub_key is None:
                    curr_ax.plot(obj[x_key], obj[y_key], color=colors[0])
                else:
                    curr_ax.plot(obj[x_key], obj[y_key][sub_key], color=colors[0])

                curr_ax.set_xlim(x_lim[0], x_lim[1])
                curr_ax.set_ylim(y_lim[0], y_lim[1])
                curr_ax.set_xticks(x_ticks)

                if idx == 0:
                    curr_ax.set_title(f"{title} LS = {obj['max_segments']}")
                    curr_ax.legend(legend)
                    curr_ax.set_ylabel(y_label)
                    curr_ax.set_xlabel(x_label)
                else:
                    curr_ax.set_title(f"LS = {obj['max_segments']}")

        for plot_coords in index_list:
            axes[plot_coords[0], plot_coords[1]].grid()
        self._save_plot(file_name=filename)

    def _plot_helper_two(self, file_name: str, erlang_indexes: list):
        data = dict()
        policy = None
        erlangs = list()
        for _, objs in self.plot_dict.items():
            for _, sim_obj in objs.items():
                erlangs = sim_obj['erlang_vals']
                policy = sim_obj['algorithm']

                data = {}
                for bandwidth, mod_objs in sim_obj['modulations'].items():
                    temp_usages = [usages for _, usages in mod_objs.items()]
                    # Flatten the multi-dimensional array
                    flat_usages = [item for sublist in temp_usages for item in sublist]
                    # Check if all usages are zero
                    if any(usage != 0 for usage in flat_usages):
                        data[bandwidth] = temp_usages

        for index in erlang_indexes:
            values_to_plot = {key: [lst[index] for lst in value] for key, value in data.items()}
            _, axis = plt.subplots(figsize=(7, 5), dpi=500)
            colors = ['orange', 'green', 'blue']
            ind = np.arange(len(values_to_plot.keys()))
            width = 0.4

            for i, key in enumerate(values_to_plot.keys()):
                for j, val in enumerate(values_to_plot[key]):
                    axis.bar(ind[i], val, width, label=key if j == 0 else "", color=colors[j],
                             bottom=sum(values_to_plot[key][:j]), edgecolor='black')

            axis.set_ylabel('Occurrences')
            axis.set_title(f"{policy} Bandwidth vs. Modulation Format Usage, E={erlangs[index]}")
            axis.set_xticks(ind)
            axis.set_xticklabels(values_to_plot.keys())
            axis.set_xlabel('Bandwidths (Gbps)')
            axis.legend(['QPSK', '16-QAM', '64-QAM'])

            axis.set_ylim(0, 17000)
            plt.tight_layout()
            values = re.findall(r'(\d+\.\d+)', policy)

            alpha, gamma = values
            self._save_plot(file_name=f"{file_name}_{alpha}_{gamma}_{erlangs[index]}")
            plt.show()

    def _plot_helper_one(self, x_vals: str, y_vals: list, file_name: str, reward_flags=None, per_req_flag=None):
        legend_list = list()
        for _, objs in self.plot_dict.items():
            for _, sim_obj in objs.items():
                # if sim_obj['algorithm'][0] == 'k':
                #     style = 'dashed'
                #     if x_vals == 'time_steps':
                #         continue
                # elif sim_obj['algorithm'][0:2] == 'Ba':
                #     style = 'dashdot'
                # else:
                style = 'solid'

                for curr_value in y_vals:
                    if reward_flags is not None:
                        for index in reward_flags[0]:
                            erlang = sim_obj['erlang_vals'][index]
                            plt.plot(sim_obj[x_vals][index], sim_obj[curr_value][reward_flags[1][0]][index],
                                     linestyle=style,
                                     markersize=2.3)
                            legend_list.append(f"E={erlang} | {sim_obj['algorithm']}")
                    elif per_req_flag is not None:
                        for index in per_req_flag:
                            erlang = sim_obj['erlang_vals'][index]
                            plt.plot(sim_obj[x_vals], sim_obj[curr_value][index],
                                     linestyle=style,
                                     markersize=2.3)
                            legend_list.append(f"E={erlang} | {sim_obj['algorithm']}")
                    else:
                        if len(y_vals) > 1:
                            legend_item = self._snake_to_title(curr_value)
                            legend_list.append(f"{legend_item}")
                        else:
                            # legend_list.append(f"{sim_obj['algorithm']}")
                            plt.plot(sim_obj[x_vals], sim_obj[curr_value], linestyle=style, markersize=2.3)

        plt.legend(legend_list)
        self._save_plot(file_name=file_name)
        plt.show()

    def plot_q_tables(self):
        plot_q_table(data=self.plot_dict)

    def plot_epsilon(self):
        self._setup_plot("Epsilon Decay vs. Time Steps", 'Epsilon', 'Time Steps (Request Numbers)',
                         y_ticks=False, x_ticks=False)
        self._plot_helper_one(x_vals='time_steps', y_vals=['epsilon'], file_name='epsilon_decay')

    def plot_td_errors(self):
        self._setup_plot("Average Errors vs. Time Steps", 'Average Error', 'Time Steps (Request Numbers)',
                         y_ticks=False, x_ticks=False)
        self._plot_helper_one(x_vals='time_steps', y_vals=['average_error'], file_name='average_errors_50',
                              reward_flags=[[0], ['routes']])

        self._setup_plot("Average Errors vs. Time Steps", 'Average Error', 'Time Steps (Request Numbers)',
                         y_ticks=False, x_ticks=False)
        self._plot_helper_one(x_vals='time_steps', y_vals=['average_error'], file_name='average_errors_300',
                              reward_flags=[[5], ['routes']])

        self._setup_plot("Average Errors vs. Time Steps", 'Average Error', 'Time Steps (Request Numbers)',
                         y_ticks=False, x_ticks=False)
        self._plot_helper_one(x_vals='time_steps', y_vals=['average_error'], file_name='average_errors_700',
                              reward_flags=[[-1], ['routes']])

    def plot_active_requests(self):
        self._setup_plot("Active Requests", 'Active Requests', 'Request Number', y_ticks=False, x_ticks=False)
        self._plot_helper_one(x_vals='req_nums', y_vals=['active_reqs'],
                              file_name='active_requests', per_req_flag=[0, 5, -1])

    def plot_network_util(self):
        self._setup_plot("Network Utilization Per Request", 'Slots Occupied', 'Request Number', y_ticks=False,
                         x_ticks=False)
        self._plot_helper_one(x_vals='req_nums', y_vals=['network_util'],
                              file_name='network_util', per_req_flag=[0, 5, -1])

    def plot_block_per_req(self):
        self._setup_plot("BP Per Request", 'Blocking Probability', 'Request Number', y_ticks=False, x_ticks=False)
        self._plot_helper_one(x_vals='req_nums', y_vals=['block_per_req'],
                              file_name='blocking_per_req', per_req_flag=[0, 5, -1])

    def plot_mod_formats(self):
        """
        Plots the modulation format usage per bandwidth.
        """
        self._plot_helper_two(file_name='mods', erlang_indexes=[0, 5, 11])

    def plot_block_reasons(self):
        """
        Plots the reasons for blocking as a percentage.
        """
        self._setup_plot("Block Reasons", 'Blocking Percentage', 'Erlang', y_ticks=False, x_ticks=False)
        self._plot_helper_one(x_vals='erlang_vals', y_vals=['cong_block', 'dist_block'],
                              file_name='block_reasons')

    def plot_rewards(self):
        """
        Plots the average rewards obtained by the agent.
        """
        self._setup_plot("Average Rewards requests", 'Average Reward', 'Time Steps (Request Numbers)',
                         y_ticks=False, x_ticks=False)
        self._plot_helper_one(x_vals='time_steps', y_vals=['average_rewards'], file_name='average_rewards_50',
                              reward_flags=[[0], ['cores']])

        self._setup_plot("Average Rewards requests", 'Average Reward', 'Time Steps (Request Numbers)',
                         y_ticks=False, x_ticks=False)
        self._plot_helper_one(x_vals='time_steps', y_vals=['average_rewards'], file_name='average_rewards_300',
                              reward_flags=[[5], ['cores']])

        self._setup_plot("Average Rewards requests", 'Average Reward', 'Time Steps (Request Numbers)',
                         y_ticks=False, x_ticks=False)
        self._plot_helper_one(x_vals='time_steps', y_vals=['average_rewards'], file_name='average_rewards_700',
                              reward_flags=[[-1], ['cores']])

    def plot_times(self):
        """
        Plots the average time of routing in milliseconds.
        """
        self._setup_plot("Average Route Time", 'Average Route Time (milliseconds)', 'Erlang', y_ticks=False)
        self._plot_helper_one(x_vals='erlang_vals', y_vals=['route_times'], file_name='average_times')

    def plot_hops(self):
        """
        Plots the average number of hops.
        """
        self._setup_plot("Average Hop Count", 'Average Hop Count', 'Erlang', y_ticks=False)
        self._plot_helper_one(x_vals='erlang_vals', y_vals=['hops'], file_name='average_hops')

    def plot_path_length(self):
        """
        Plots the average path length.
        """
        self._setup_plot("Average Path Length", 'Average Path Length (KM)', 'Erlang', y_ticks=False)
        self._plot_helper_one(x_vals='erlang_vals', y_vals=['path_lengths'], file_name='average_lengths')

    def plot_blocking(self):
        """
        Plots the average blocking probability for each Erlang value.
        """
        self._setup_plot("Average Blocking Prob. vs. Erlang", 'Average Blocking Probability', 'Erlang', y_ticks=True)
        self._plot_helper_one(x_vals='erlang_vals', y_vals=['blocking_vals'], file_name='average_bp')


def _and_filters(filters, content):
    keep_config = True
    content = content.get('sim_params')
    for check_flags in filters['and_filters']:
        search_keys = check_flags[0:-1]
        check_value = check_flags[-1]

        for curr_key in search_keys:
            content = content.get(curr_key)

        if content != check_value:
            keep_config = False
            break

    return keep_config


def _or_filters(filters, content):
    keep_config = True
    for sub_flags in filters['or_filters']:
        check_content = content.get('sim_params')
        search_keys = sub_flags[0:-1]
        check_value = sub_flags[-1]

        for curr_key in search_keys:
            check_content = check_content.get(curr_key)

        if check_content == check_value:
            keep_config = True
            break

        keep_config = False

    return keep_config


def _not_filters(filters, content):
    keep_config = True
    for sub_flags in filters['not_filters']:
        check_content = content.get('sim_params')
        search_keys = sub_flags[0:-1]
        check_value = sub_flags[-1]

        for curr_key in search_keys:
            check_content = check_content.get(curr_key)

        if check_content == check_value:
            keep_config = False
            break

        keep_config = True

    return keep_config


def _check_filters(file_content, filters):
    keep_config = _and_filters(filters=filters, content=file_content)

    if keep_config:
        keep_config = _or_filters(filters=filters, content=file_content)

        if keep_config:
            keep_config = _not_filters(filters=filters, content=file_content)

    return keep_config


def find_times(dates_networks: dict, filters: dict):
    """
    Given a list of dates, find simulations based on given parameters being the same. For example, each simulation
    should have 256 spectral slots.

    :param dates_networks: The desired dates and networks to sort through.
    :type dates_networks: dict

    :param filters: The parameters for filtering.
    :type filters: dict

    :return: Every time and simulation number is filtered by date and given params.
    :rtype: dict
    """
    resp_times = list()
    resp_sims = list()
    resp_networks = list()
    resp_dates = list()
    info_dict = dict()
    for curr_date, curr_network in dates_networks.items():
        times_path = f'../data/output/{curr_network}/{curr_date}'
        sim_times = [d for d in os.listdir(times_path) if os.path.isdir(os.path.join(times_path, d))]

        for curr_time in sim_times:
            sims_path = f'{times_path}/{curr_time}'
            sim_numbers = [d for d in os.listdir(sims_path) if os.path.isdir(os.path.join(sims_path, d))]

            for curr_sim in sim_numbers:
                files_path = f'{sims_path}/{curr_sim}'
                files = [f for f in os.listdir(files_path) if os.path.isfile(os.path.join(files_path, f))]
                # All Erlangs will have the same configuration, just take the first file
                des_file = files[0]
                with open(f'{files_path}/{des_file}', 'r', encoding='utf-8') as file_obj:
                    file_content = json.load(file_obj)

                keep_config = _check_filters(file_content=file_content, filters=filters)
                if keep_config:
                    if curr_time not in info_dict:
                        info_dict[curr_time] = {'sim_nums': list(), 'networks': list(), 'dates': list()}
                    info_dict[curr_time]['sim_nums'].append(curr_sim)
                    info_dict[curr_time]['networks'].append(curr_network)
                    info_dict[curr_time]['dates'].append(curr_date)

    for time, obj in info_dict.items():
        resp_times.append([time])
        resp_sims.append(obj['sim_nums'])
        resp_networks.append(obj['networks'])
        resp_dates.append(obj['dates'])

    return resp_times, resp_sims, resp_networks, resp_dates


def main():
    """
    Controls this script.
    """
    filters = {
        'and_filters': [
            # ['ai_arguments', 'policy', 'policy_one']
        ],
        'or_filters': [
            # ['ai_arguments', 'policy', 'policy_one'],
            # ['route_method', 'k_shortest_path'],
        ],
        'not_filters': [
            # ['route_method', 'k_shortest_path']
        ]
    }
    sim_times, sim_nums, networks, dates = find_times(dates_networks={'0118': 'USNet'}, filters=filters)
    plot_obj = PlotStats(net_names=networks, dates=dates, times=sim_times, sims=sim_nums)

    plot_obj.plot_blocking()
    # plot_obj.plot_path_length()
    # plot_obj.plot_hops()
    # plot_obj.plot_rewards()
    # plot_obj.plot_block_reasons()
    # plot_obj.plot_mod_formats()
    # plot_obj.plot_block_per_req()
    # plot_obj.plot_network_util()
    # plot_obj.plot_active_requests()
    # plot_obj.plot_td_errors()
    # plot_obj.plot_epsilon()
    # plot_obj.plot_q_tables()


if __name__ == '__main__':
    main()
