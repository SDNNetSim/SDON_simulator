# Standard library imports
import os
import json

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np

# Local application imports
from useful_functions.handle_dirs_files import create_dir


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
        # Miscellaneous customizations visually
        self.colors = ['#024de3', '#00b300', 'orange', '#6804cc', '#e30220']
        self.line_styles = ['solid', 'dashed', 'dotted', 'dashdot']
        self.markers = ['o', '^', 's', 'x']
        self.x_ticks = [200, 300, 400, 500, 600, 700]

        self._get_data()

    def _get_file_info(self):
        resp = dict()
        for network, date, times in zip(self.net_names, self.dates, self.times):
            for curr_time in times:
                resp[curr_time] = {'network': network, 'date': date, 'sims': dict()}
                curr_dir = f'{self.base_dir}/{network}/{date}/{curr_time}'

                # Sort by sim number
                sim_dirs = os.listdir(curr_dir)
                sim_dirs = sorted(sim_dirs, key=lambda x: int(x[1:]))

                for curr_sim in sim_dirs:
                    # User selected to not run this simulation
                    if curr_sim not in self.sims:
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
        title = ", ".join(input_list[:-1]) + " & " + input_list[-1] if len(input_list) > 1 else input_list[0]
        return title

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

    def _find_algorithm(self):
        route_method = self._snake_to_title(snake_str=self.erlang_dict['sim_params']['route_method'])
        # TODO: Use eventually
        # alloc_method = self._snake_to_title(snake_str=self.erlang_dict['sim_params']['allocation_method'])

        if route_method == 'Xt Aware':
            param = f"B = {self.erlang_dict['sim_params']['beta']}"
        else:
            param = f"k = {self.erlang_dict['sim_params']['k_paths']}"

        algorithm = f'{param}'
        self.plot_dict[self.time][self.sim_num]['algorithm'] = algorithm

    def _find_sim_info(self, network):
        self.plot_dict[self.time][self.sim_num]['holding_time'] = self.erlang_dict['sim_params']['holding_time']
        self.plot_dict[self.time][self.sim_num]['cores_per_link'] = self.erlang_dict['sim_params']['cores_per_link']
        self.plot_dict[self.time][self.sim_num]['spectral_slots'] = self.erlang_dict['sim_params']['spectral_slots']
        self.plot_dict[self.time][self.sim_num]['network'] = network

        self.num_requests = self._int_to_string(self.erlang_dict['sim_params']['num_requests'])
        self.num_cores = self.erlang_dict['sim_params']['cores_per_link']

    def _find_misc_stats(self):
        path_lens = self._dict_to_list(self.erlang_dict['misc_stats'], 'path_lengths')
        hops = self._dict_to_list(self.erlang_dict['misc_stats'], 'hops')
        times = self._dict_to_list(self.erlang_dict['misc_stats'], 'route_times') * 10 ** 3

        average_len = np.mean(path_lens)
        average_hop = np.mean(hops)
        average_times = np.mean(times)

        self.plot_dict[self.time][self.sim_num]['path_lengths'].append(average_len)
        self.plot_dict[self.time][self.sim_num]['hops'].append(average_hop)
        self.plot_dict[self.time][self.sim_num]['route_times'].append(average_times)

    def _find_blocking(self):
        blocking_mean = self.erlang_dict['blocking_mean']
        self.plot_dict[self.time][self.sim_num]['blocking_vals'].append(blocking_mean)

    def _find_crosstalk(self):
        xt_vals = self._dict_to_list(self.erlang_dict['misc_stats'], 'weight_mean')
        average_xt = np.mean(xt_vals)
        std_xt = self._dict_to_list(self.erlang_dict['misc_stats'], 'weight_std')
        min_xt = xt_vals.min(initial=np.inf)

        self.plot_dict[self.time][self.sim_num]['xt_vals'].append(average_xt)
        self.plot_dict[self.time][self.sim_num]['xt_std'].append(std_xt)
        self.plot_dict[self.time][self.sim_num]['min_xt_vals'].append(min_xt)

    def _update_plot_dict(self):
        if self.plot_dict is None:
            self.plot_dict = {self.time: {}}
        else:
            self.plot_dict[self.time] = {}

        self.plot_dict[self.time][self.sim_num] = {
            'erlang_vals': [],
            'blocking_vals': [],
            'xt_vals': [],
            'min_xt_vals': [],
            'xt_std': [],
            'path_lengths': [],
            'hops': [],
            'route_times': [],
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
                    curr_fp = os.path.join(self.base_dir, obj['network'], obj['date'], time, curr_sim,
                                           f'{erlang}_erlang.json')
                    with open(curr_fp, 'r', encoding='utf-8') as file_obj:
                        erlang_dict = json.load(file_obj)

                    erlang = int(erlang.split('.')[0])
                    self.plot_dict[self.time][self.sim_num]['erlang_vals'].append(erlang)

                    self.erlang_dict = erlang_dict
                    self._find_blocking()
                    self._find_crosstalk()
                    self._find_algorithm()
                    self._find_misc_stats()
                    self._find_sim_info(network=obj['network'])

    def _save_plot(self, file_name: str):
        file_path = f'./output/{self.net_names[-1]}/{self.dates[-1]}/{self.times[0][-1]}'
        create_dir(file_path)
        plt.savefig(f'{file_path}/{file_name}_core{self.num_cores}.png')

    def _setup_plot(self, title, y_label, x_label, grid=True, y_ticks=True, y_lim=False, x_ticks=True):
        plt.figure(figsize=(7, 5), dpi=300)
        plt.title(title)
        plt.ylabel(y_label)
        plt.xlabel(x_label)

        if y_ticks:
            plt.yticks([10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1])
            plt.ylim(10 ** -5, 1)
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

    def _plot_helper_one(self, x_vals: str, y_vals: str, file_name: str):
        legend_list = list()
        for _, objs in self.plot_dict.items():
            for _, sim_obj in objs.items():
                if sim_obj['algorithm'][0] == 'B':
                    style = 'solid'
                else:
                    style = 'dashed'

                plt.plot(sim_obj[x_vals], sim_obj[y_vals], linestyle=style, markersize=2.3)
                legend_list.append(f"{sim_obj['algorithm']}")

        plt.legend(legend_list)
        self._save_plot(file_name=file_name)
        plt.show()

    def plot_times(self):
        """
        Plots the average time of routing in milliseconds.
        """
        network = self._list_to_title(input_list=self.net_names)
        self._setup_plot(f"{network} Average Route Time {self.num_requests} requests (C={self.num_cores})",
                         'Average Route Time (milliseconds)', 'Erlang', y_ticks=False, y_lim=(0, 65))
        self._plot_helper_one(x_vals='erlang_vals', y_vals='route_times', file_name='average_times')

    def plot_hops(self):
        """
        Plots the average number of hops.
        """
        network = self._list_to_title(input_list=self.net_names)
        self._setup_plot(f"{network} Average Hop Count {self.num_requests} requests (C={self.num_cores})",
                         'Average Hop Count', 'Erlang', y_ticks=False, y_lim=(2.0, 2.45))
        self._plot_helper_one(x_vals='erlang_vals', y_vals='hops', file_name='average_hops')

    def plot_path_length(self):
        """
        Plots the average path length.
        """
        network = self._list_to_title(input_list=self.net_names)
        self._setup_plot(f"{network} Average Path Length {self.num_requests} requests (C={self.num_cores})",
                         'Average Path Length (KM)', 'Erlang', y_ticks=False)
        self._plot_helper_one(x_vals='erlang_vals', y_vals='path_lengths', file_name='average_lengths')

    def plot_crosstalk(self):
        """
        Plots the average XT values.
        """
        network = self._list_to_title(input_list=self.net_names)
        self._setup_plot(f'{network} Average XT vs. Erlang {self.num_requests} requests (C={self.num_cores})',
                         'Crosstalk (dB)', 'Erlang', y_ticks=False, y_lim=(-33.75, -32.0))

        self._plot_helper_one(x_vals='erlang_vals', y_vals='xt_vals', file_name='average_xt')

        self._setup_plot(f'{network} Min XT vs. Erlang {self.num_requests} requests (C={self.num_cores})',
                         'Crosstalk (dB)', 'Erlang', y_ticks=False, y_lim=(-33.75, -32.0))

        self._plot_helper_one(x_vals='erlang_vals', y_vals='min_xt_vals', file_name='min_xt')

    def plot_blocking(self):
        """
        Plots the average blocking probability for each Erlang value.
        """
        network = self._list_to_title(input_list=self.net_names)
        self._setup_plot(f"{network} Average Blocking Prob. {self.num_requests} requests (C={self.num_cores})",
                         'Average Blocking Probability', 'Erlang')
        self._plot_helper_one(x_vals='erlang_vals', y_vals='blocking_vals', file_name='average_bp')


def main():
    """
    Controls this script.
    """
    times = [
        # beta = 1.0
        '12_25_44_950876',
        # beta = 0.9
        '12_25_44_727252',
        # beta = 0.8
        '12_25_13_633317',
        # beta = 0.7
        '11_35_03_895500',
        # beta = 0.6
        '11_35_01_021682',
        # beta = 0.5
        '22_38_06_455877',
        # beta = 0.4
        '22_38_06_317002',
        # beta = 0.3
        '22_38_06_348728',
        # beta = 0.2
        '22_38_06_348645',
        # beta = 0.1
        '22_38_06_348798',
        # beta = 0.000001
        '22_38_06_397731',
        # k = 2
        '16_15_55_029251',
        # k = 3
        '16_15_52_100304',
        # k = 4
        '16_15_52_280391',
        # k = 5
        '16_15_53_848492',
    ]
    plot_obj = PlotStats(net_names=['NSFNet'], dates=['0930'], times=[times],
                         sims=['s1'])
    # plot_obj.plot_blocking()
    plot_obj.plot_crosstalk()
    # plot_obj.plot_path_length()
    # plot_obj.plot_hops()
    # plot_obj.plot_times()


if __name__ == '__main__':
    main()
