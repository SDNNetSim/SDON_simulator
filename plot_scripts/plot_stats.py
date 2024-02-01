import os
import re

import matplotlib.pyplot as plt
import numpy as np

from arg_scripts.plot_args import empty_props
from helper_scripts.plot_helpers import PlotHelpers, find_times


class PlotStats:
    """
    A class for computing and plotting statistical analysis for simulations.
    """

    def __init__(self, sims_info_dict: dict):
        self.plot_props = empty_props
        self.plot_help_obj = PlotHelpers(plot_props=self.plot_props)

        self.file_info = self.plot_help_obj.get_file_info(sims_info_dict=sims_info_dict)

    def _save_plot(self, file_name: str):
        pass
        # file_path = f'./output/{self.net_names[0][-1]}/{self.dates[0][-1]}/{self.plot_props['time']s[0][-1]}'
        # create_dir(file_path)
        # plt.savefig(f'{file_path}/{file_name}_core{self.num_cores}.png')

    def _setup_plot(self, title, y_label, x_label, grid=True, y_ticks=True, y_lim=False, x_ticks=True):
        plt.figure(figsize=(7, 5), dpi=300)
        plt.title(f"{self.title_names} {title} R={self.num_requests}")
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

    # TODO: Transfer to a loop or something creative
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
