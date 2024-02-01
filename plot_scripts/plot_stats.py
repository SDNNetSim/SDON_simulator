import re
import os

import matplotlib.pyplot as plt
import numpy as np

from arg_scripts.plot_args import empty_props
from helper_scripts.os_helpers import create_dir
from helper_scripts.plot_helpers import PlotHelpers, find_times


# TODO: Double check this script and all scripts tomorrow
# TODO: Add number of cores and number of requests
class PlotStats:
    """
    A class for computing and plotting statistical analysis for simulations.
    """

    def __init__(self, sims_info_dict: dict):
        self.plot_props = empty_props
        self.sims_info_dict = sims_info_dict
        self.plot_help_obj = PlotHelpers(plot_props=self.plot_props, net_names_list=sims_info_dict['networks_matrix'])

        self.plot_help_obj.get_file_info(sims_info_dict=sims_info_dict)

    def _save_plot(self, file_name: str):
        # Default to the earliest time for saving
        time = self.sims_info_dict['times_matrix'][0][-1]
        network = self.sims_info_dict['networks_matrix'][0][-1]
        date = self.sims_info_dict['dates_matrix'][0][-1]
        save_fp = os.path.join('..', 'data', 'plots', network, date, time)
        create_dir(file_path=save_fp)

        save_fp = os.path.join(save_fp, file_name)
        plt.savefig(save_fp)

    def _setup_plot(self, title, y_label, x_label, grid=True, y_ticks=True, y_lim=False, x_ticks=True):
        plt.figure(figsize=(7, 5), dpi=300)
        plt.title(f"{self.plot_props['title_names']} {title}")
        plt.ylabel(y_label)
        plt.xlabel(x_label)

        if y_ticks:
            plt.yticks([10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1])
            plt.ylim(10 ** -5, 1)
            plt.yscale('log')

        if y_lim:
            plt.ylim(y_lim[0], y_lim[1])

        if x_ticks:
            plt.xticks(self.plot_props['x_tick_list'])
            plt.xlim(self.plot_props['x_tick_list'][0], self.plot_props['x_tick_list'][-1])

        if grid:
            plt.grid()

    # TODO: Fix this up
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

    def _plot_helper_one(self, x_vals: str, y_vals: str, file_name: str):
        legend_list = list()
        color_count = 0
        style_count = 0
        for _, sims_dict in self.plot_props['plot_dict'].items():
            for _, info_dict in sims_dict.items():
                color = self.plot_props['color_list'][color_count]
                style = self.plot_props['style_list'][style_count]
                plt.plot(info_dict[x_vals], info_dict[y_vals], linestyle=style, markersize=2.3, color=color)
                legend_list.append(color)
                color_count += 1

            color_count = 0
            style_count += 1

        plt.legend(legend_list)
        self._save_plot(file_name=file_name)
        plt.show()

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
        self._plot_helper_one(x_vals='erlang_list', y_vals='blocking_list', file_name='average_bp.png')


def main():
    """
    Controls this script.
    """
    filter_dict = {
        'and_filter_list': [
        ],
        'or_filter_list': [
        ],
        'not_filter_list': [
            # ['route_method', 'k_shortest_path']
        ]
    }

    sims_info_dict = find_times(dates_dict={'0201': 'USNet'}, filter_dict=filter_dict)
    plot_obj = PlotStats(sims_info_dict=sims_info_dict)

    plot_obj.plot_blocking()
    # plot_obj.plot_path_length()
    # plot_obj.plot_hops()
    # plot_obj.plot_block_reasons()
    # plot_obj.plot_mod_formats()
    # plot_obj.plot_block_per_req()
    # plot_obj.plot_network_util()
    # plot_obj.plot_active_requests()


if __name__ == '__main__':
    main()
