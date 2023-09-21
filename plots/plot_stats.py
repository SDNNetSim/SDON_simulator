# Standard library imports
import os
import json

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np

# Local application imports
from useful_functions.handle_dirs_files import create_dir


# TODO: Fix other methods to work for new structure

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
        self.file_info = self.get_file_info()

        # The final dictionary containing information for all plots
        self.plot_dict = None
        self.time = None
        self.sim_num = None
        self.erlang_dict = None
        self.num_cores = None
        # Miscellaneous customizations visually
        self.colors = ['#024de3', '#00b300', 'orange', '#6804cc', '#e30220']
        self.line_styles = ['solid', 'dashed', 'dotted', 'dashdot']
        self.markers = ['o', '^', 's', 'x']
        self.x_ticks = [800, 850, 900, 950, 1000]

        self.get_data()

    def get_file_info(self):
        """
        Obtains all the filenames of the output data from the simulations.
        """
        resp = dict()
        # Default to the latest time if a data directory wasn't specified
        if self.dates is None or self.times is None:
            dirs = [f'{self.base_dir}/{self.net_names[0]}/']
            dir_list = os.listdir(dirs[0])
            self.dates = [sorted(dir_list)[-1]]
            dir_list = os.listdir(os.path.join(dirs[0], self.dates[0]))
            self.times = [sorted(dir_list)[-1]]
            dirs = [os.path.join(dirs[0], f'{self.dates[0]}/{self.times[0]}')]

        for network, date, time in zip(self.net_names, self.dates, self.times):
            resp[time] = {'network': network, 'date': date, 'sims': dict()}
            curr_dir = f'{self.base_dir}/{network}/{date}/{time}'

            # Sort by sim number
            sim_dirs = os.listdir(curr_dir)
            sim_dirs = sorted(sim_dirs, key=lambda x: int(x[1:]))

            for curr_sim in sim_dirs:
                # User selected to not run this simulation
                if curr_sim not in self.sims:
                    continue
                curr_fp = os.path.join(curr_dir, curr_sim)
                resp[time]['sims'][curr_sim] = list()
                files = os.listdir(curr_fp)
                sorted_files = sorted(files, key=lambda x: float(x.split('_')[0]))

                for erlang_file in sorted_files:
                    resp[time]['sims'][curr_sim].append(erlang_file.split('_')[0])

        return resp

    def _find_misc_info(self, erlang):
        # TODO: Params for this
        if erlang in (10, 100, 400):
            self.plot_dict[self.time][self.sim_num]['taken_slots'][erlang] = dict()
            self.plot_dict[self.time][self.sim_num]['block_per_req'][erlang] = dict()
            self.plot_dict[self.time][self.sim_num]['active_requests'][erlang] = dict()
            self.plot_dict[self.time][self.sim_num]['guard_bands'][erlang] = dict()
            self.plot_dict[self.time][self.sim_num]['num_segments'][erlang] = list()

            for request_number, request_info in self.erlang_dict['misc_stats']['request_snapshots'].items():
                request_number = int(request_number)
                # Plot every 'x' amount of request snapshots
                if request_number % 1 == 0 or request_number == 1:
                    cores_per_link = self.erlang_dict['misc_stats']['cores_per_link']

                    taken_slots = request_info['occ_slots'] / cores_per_link
                    self.plot_dict[self.time][self.sim_num]['taken_slots'][erlang][request_number] = taken_slots
                    self.plot_dict[self.time][self.sim_num]['block_per_req'][erlang][request_number] = request_info[
                        'blocking_prob']

                    active_requests = request_info['active_requests'] / cores_per_link
                    self.plot_dict[self.time][self.sim_num]['active_requests'][erlang][request_number] = active_requests

                    guard_bands = request_info['guard_bands'] / cores_per_link
                    self.plot_dict[self.time][self.sim_num]['guard_bands'][erlang][request_number] = guard_bands

                self.plot_dict[self.time][self.sim_num]['num_segments'][erlang].append(request_info['num_segments'])

    @staticmethod
    def _snake_to_title(snake_str):
        words = snake_str.split('_')
        title_str = ' '.join(word.capitalize() for word in words)
        return title_str

    def _find_algorithm(self):
        route_method = self._snake_to_title(snake_str=self.erlang_dict['misc_stats']['route_method'])
        alloc_method = self._snake_to_title(snake_str=self.erlang_dict['misc_stats']['alloc_method'])

        algorithm = f'{route_method} - {alloc_method}'
        self.plot_dict[self.time][self.sim_num]['algorithm'] = algorithm

    def _find_sim_info(self, network):
        self.plot_dict[self.time][self.sim_num]['hold_time_mean'] = self.erlang_dict['misc_stats']['hold_time_mean']
        self.plot_dict[self.time][self.sim_num]['cores_per_link'] = self.erlang_dict['misc_stats']['cores_per_link']
        self.plot_dict[self.time][self.sim_num]['spectral_slots'] = self.erlang_dict['misc_stats']['spectral_slots']
        self.plot_dict[self.time][self.sim_num]['network'] = network

        self.plot_dict[self.time][self.sim_num]['max_segments'] = self.erlang_dict['misc_stats']['max_segments']
        self.num_cores = self.erlang_dict['misc_stats']['cores_per_link']

    def _find_blocking(self):
        # For training, no mean was calculated as we wanted to run for every iteration (no CI)
        if self.erlang_dict['misc_stats']['is_training']:
            self.plot_dict[self.time][self.sim_num]['blocking_vals'].append(
                np.average(list(self.erlang_dict['block_per_sim'].values())))
        else:
            self.plot_dict[self.time][self.sim_num]['blocking_vals'].append(
                self.erlang_dict['misc_stats']['blocking_mean'])

        self.plot_dict[self.time][self.sim_num]['distance_block'].append(self.erlang_dict['misc_stats']['dist_percent'])
        self.plot_dict[self.time][self.sim_num]['cong_block'].append(self.erlang_dict['misc_stats']['cong_percent'])

    def _find_bw_blocks(self):
        # Convert how many times a bandwidth was blocked to a percentage
        tmp_sum = sum(self.erlang_dict['misc_stats']['block_per_bw'].values())
        for key in self.erlang_dict['misc_stats']['block_per_bw']:
            try:
                tmp_value = self.erlang_dict['misc_stats']['block_per_bw'][key] / tmp_sum * 100
            except ZeroDivisionError:
                tmp_value = 0

            if key not in self.plot_dict[self.time][self.sim_num]['block_per_bw'].keys():
                self.plot_dict[self.time][self.sim_num]['block_per_bw'][key] = list()

            self.plot_dict[self.time][self.sim_num]['block_per_bw'][key].append(round(tmp_value, 2))

    def _update_plot_dict(self):
        if self.plot_dict is None:
            self.plot_dict = {self.time: {}}

        self.plot_dict[self.time][self.sim_num] = {
            'erlang_vals': [],
            'blocking_vals': [],
            'average_transponders': [],
            'distance_block': [],
            'cong_block': [],
            'block_per_bw': {},
            'hold_time_mean': None,
            'cores_per_link': None,
            'spectral_slots': None,
            'max_segments': None,
            'num_segments': {},
            'taken_slots': {},
            'block_per_req': {},
            'active_requests': {},
            'guard_bands': {},
        }

    def get_data(self):
        """
        Structures all data to be plotted.
        """
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
                    self.plot_dict[self.time][self.sim_num]['average_transponders'].append(
                        erlang_dict['misc_stats']['trans_mean'])

                    self.erlang_dict = erlang_dict
                    self._find_blocking()
                    self._find_bw_blocks()
                    self._find_algorithm()
                    self._find_sim_info(network=obj['network'])
                    self._find_misc_info(erlang=erlang)

    @staticmethod
    def running_average(lst, interval):
        """
        Given a list, takes a running average of that list and appends at specific intervals.

        :param lst: A list of floats or ints
        :type lst: list

        :param interval: Determines the intervals for the running averages
        :type interval: int

        :return: A list of request numbers for specific intervals and the running averages up until each interval
        """
        request_numbers = list()
        averages = list()

        run_sum = 0
        for i, value in enumerate(lst, start=1):
            run_sum += value

            if i % interval == 0:
                request_numbers.append(i)
                averages.append(run_sum / i)

        return request_numbers, averages

    def _save_plot(self, file_name):
        """
        Save a Matplotlib plot to a file with the specified name.

        :param file_name: A string representing the name of the file to save the plot as
        :type file_name: str
        """
        file_path = f'./output/{self.net_names}/{self.dates}/{self.times}'
        create_dir(file_path)
        plt.savefig(f'{file_path}/{file_name}_core{self.num_cores}.png')

    def _setup_plot(self, title, y_label, x_label, grid=True, y_ticks=True, x_ticks=True):
        """
        Set up a Matplotlib plot with a given title, y-axis label, and x-axis label.

        :param title: A string representing the title of the plot.
        :type title: str

        :param y_label: A string representing the label for the y-axis of the plot.
        :type y_label: str

        :param x_label: A string representing the label for the x-axis of the plot.
        :type x_label: str

        :param grid: A boolean indicating whether to show a grid on the plot.
        :type grid: bool

        :param y_ticks: Determines if we'd like to plot the default for y_ticks, which is blocking probability
                        (log scale)
        :type y_ticks: bool

        :param x_ticks: Determines if we'd like to plot the default for x_ticks, which are Erlang values
        :type x_ticks: bool
        """
        plt.figure(figsize=(7, 5), dpi=300)
        plt.title(title)
        plt.ylabel(y_label)
        plt.xlabel(x_label)

        if y_ticks:
            plt.ylim(10 ** -5, 1)
            plt.yscale('log')

        if x_ticks:
            plt.xticks(self.x_ticks)
            plt.xlim(self.x_ticks[0], self.x_ticks[-1])

        if grid:
            plt.grid()

    def _plot_sub_plots(self, plot_info, title, x_ticks, x_lim, x_label, y_ticks, y_lim, y_label, legend, filename):
        """
        Plot four by four Matplotlib subplots.
        """
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

    def plot_blocking(self):
        """
        Plots the blocking probability for each Erlang value.
        """
        self._setup_plot('USNet Average BP vs. Erlang (C=7)', 'Blocking Probability', 'Erlang')

        style_count = 0
        legend_list = list()
        for _, objs in self.plot_dict.items():
            for _, sim_obj in objs.items():
                color = self.colors[style_count]
                line_style = self.line_styles[style_count]
                marker = self.markers[style_count]

                plt.plot(sim_obj['erlang_vals'], sim_obj['blocking_vals'], color=color, linestyle=line_style,
                         marker=marker, markersize=2.3)
                legend_list.append(f"{sim_obj['algorithm']}")

                style_count += 1

        plt.yticks([10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1])
        plt.legend(legend_list)

        self._save_plot(file_name='blocking')
        plt.show()

    def plot_blocking_per_request(self):
        """
        Plots the blocking probability, but for each request at that point in time.
        """
        self._setup_plot(f'{self.net_names} Request Snapshot vs. Blocking Probability (C={self.num_cores})',
                         'Blocking Probability', 'Request Number', x_ticks=False)

        legend_list = list()
        style_count = 0
        marker_count = 1
        for _, thread_obj in self.plot_dict.items():
            color = self.colors[style_count]

            for erlang in thread_obj['block_per_req']:
                lst = list(thread_obj['block_per_req'][erlang].values())
                request_numbers, slots_occupied = self.running_average(lst=lst, interval=500)
                marker = self.markers[marker_count]

                plt.plot(request_numbers, slots_occupied, color=color, marker=marker, markersize=2.3)

                legend_list.append(f"E={erlang} LS={thread_obj['max_segments']}")
                marker_count += 1

            marker_count = 1
            style_count += 1

        plt.legend(legend_list)
        plt.xlim(1000, 10000)
        self._save_plot(file_name='block_per_request')
        plt.show()

    def plot_transponders(self):
        """
        Plots the average number of transponders used for each Erlang value.
        """
        self._setup_plot(f'{self.net_names} Transponders vs. Erlang (C={self.num_cores})', 'Transponders', 'Erlang',
                         y_ticks=False)
        plt.ylim(0.9, 2.5)

        legend_list = list()
        style_count = 0
        for thread, thread_obj in self.plot_dict.items():  # pylint: disable=unused-variable
            color = self.colors[style_count]

            plt.plot(thread_obj['erlang_vals'], thread_obj['average_transponders'], color=color)
            legend_list.append(f"C={thread_obj['cores_per_link']} LS={thread_obj['max_segments']}")

            style_count += 1

        plt.legend(legend_list)
        self._save_plot(file_name='transponders')
        plt.show()

    def plot_slots_taken(self):
        """
        Plots the number of slots taken in the entire network for certain request snapshots.
        """
        self._setup_plot(f'{self.net_names} Request Snapshot vs. Slots Occupied (C={self.num_cores})', 'Slots Occupied',
                         'Request Number',
                         y_ticks=False, x_ticks=False)

        legend_list = list()
        style_count = 0
        marker_count = 1
        for _, thread_obj in self.plot_dict.items():
            color = self.colors[style_count]
            for erlang in thread_obj['taken_slots']:
                lst = list(thread_obj['taken_slots'][erlang].values())
                request_numbers, slots_occupied = self.running_average(lst=lst, interval=500)
                marker = self.markers[marker_count]

                plt.plot(request_numbers, slots_occupied, color=color, marker=marker, markersize=2.3)

                legend_list.append(f"E={erlang} LS={thread_obj['max_segments']}")
                marker_count += 1

            marker_count = 1
            style_count += 1

        plt.legend(legend_list, loc='upper left')
        plt.xlim(1000, 10000)
        plt.ylim(0, 4100)
        self._save_plot(file_name='slots_occupied')
        plt.show()

    # TODO: Bugfix needed? Some LS are not equal to 1, 2, 4, or 8
    def plot_num_segments(self):
        """
        Plots the number of segments each request has been sliced into.
        """
        self._setup_plot(f'{self.net_names} Number of Segments vs. Occurrences (C={self.num_cores})', 'Occurrences',
                         'Number of Segments',
                         y_ticks=False, x_ticks=False, grid=False)

        erlang_colors = ['#0000b3', '#3333ff', '#9999ff', '#b30000', '#ff3333', '#ff9999', '#00b33c', '#00ff55',
                         '#99ffbb']

        hist_list = list()
        legend_list = list()
        for _, thread_obj in self.plot_dict.items():
            # No slicing will occur
            if thread_obj['max_segments'] == 1:
                continue
            for erlang, slices_lst in thread_obj['num_segments'].items():
                hist_list.append(slices_lst)
                legend_list.append(f"E={int(erlang)} LS={thread_obj['max_segments']}")

        bins = [1, 2, 3, 4, 5, 6, 7, 8]
        plt.hist(hist_list, stacked=False, histtype='bar', edgecolor='black', rwidth=1, color=erlang_colors, bins=bins)

        plt.ylim(0, 1000)
        plt.xlim(0, 8)
        plt.legend(legend_list, loc='upper right')
        self._save_plot(file_name='num_segments')
        plt.show()

    def plot_active_requests(self):
        """
        Plots the amount of active requests in the network at specific intervals.
        """
        self._setup_plot(f'{self.net_names} Request Snapshot vs. Active Requests (C={self.num_cores})',
                         'Active Requests', 'Request Number',
                         y_ticks=False, x_ticks=False)

        legend_list = list()
        style_count = 0
        marker_count = 1
        for _, thread_obj in self.plot_dict.items():
            color = self.colors[style_count]
            for erlang in thread_obj['active_requests']:
                request_numbers = [key for key in thread_obj['active_requests'][erlang].keys() if
                                   key % 500 == 0]

                active_requests = [thread_obj['active_requests'][erlang][index] for index in request_numbers]

                marker = self.markers[marker_count]
                plt.plot(request_numbers, active_requests, color=color, marker=marker, markersize=2.3)

                legend_list.append(f"E={erlang} LS={thread_obj['max_segments']}")
                marker_count += 1

            marker_count = 1
            style_count += 1

        plt.legend(legend_list, loc='upper left')
        plt.xlim(1000, 10000)
        plt.ylim(0, 400)
        self._save_plot(file_name='active_requests')
        plt.show()

    def plot_guard_bands(self):
        """
        Plots the number of guard bands being used in the network on average.
        """
        self._setup_plot(f'{self.net_names} Request Snapshot vs. Guard Bands (C={self.num_cores})', 'Guard Bands',
                         'Request Number',
                         y_ticks=False, x_ticks=False)

        legend_list = list()
        style_count = 0
        marker_count = 1
        for _, thread_obj in self.plot_dict.items():
            color = self.colors[style_count]
            for erlang in thread_obj['guard_bands']:
                lst = list(thread_obj['guard_bands'][erlang].values())
                request_numbers, guard_bands = self.running_average(lst=lst, interval=50)

                marker = self.markers[marker_count]
                plt.plot(request_numbers, guard_bands, color=color, marker=marker, markersize=2.3)

                legend_list.append(f"E={erlang} LS={thread_obj['max_segments']}")
                marker_count += 1

            marker_count = 1
            style_count += 1

        plt.legend(legend_list, loc='upper left')
        plt.xlim(50, 1000)
        plt.ylim(0, 800)
        self._save_plot(file_name='guard_bands')
        plt.show()

    def plot_dist_cong(self):
        """
        Plot the blocking percentages broken down into distance vs. congestion.
        """
        cong_colors = {0: '#800000'}
        dist_colors = {0: '#009900'}
        plot_info = [
            ('erlang_vals', 'cong_block', None, cong_colors),
            ('erlang_vals', 'distance_block', None, dist_colors)
        ]

        self._plot_sub_plots(plot_info=plot_info, x_ticks=self.x_ticks, x_lim=[self.x_ticks[0], self.x_ticks[-1]],
                             x_label='Erlang', y_ticks=[20, 40, 60, 80, 100], y_lim=[-1, 101], y_label='Percent',
                             legend=['Cong.', 'Dist.'], filename='percents', title=f"Block Percent {self.net_names}")
        plt.show()

    def plot_bandwidths(self):
        """
        Plot the blocking percentages broken down by bandwidth.
        """
        band_colors_one = {0: '#005DCC'}
        band_colors_two = {0: '#00CC00'}
        band_colors_three = {0: '#CC0000'}
        plot_info = [
            ('erlang_vals', 'block_per_bw', '50', band_colors_one),
            ('erlang_vals', 'block_per_bw', '100', band_colors_two),
            ('erlang_vals', 'block_per_bw', '400', band_colors_three),
        ]

        self._plot_sub_plots(plot_info=plot_info, x_ticks=self.x_ticks, x_lim=[self.x_ticks[0], self.x_ticks[-1]],
                             x_label='Erlang', y_ticks=[20, 40, 60, 80, 100], y_lim=[-1, 101], y_label='Percent',
                             legend=['50', '100', '400'], filename='bandwidths', title=f'Band Block {self.net_names}')
        plt.show()


def main():
    """
    Controls this script.
    """
    plot_obj = PlotStats(net_names=['USNet', 'USNet'], dates=['0921'], times=['19:32:30'],
                         sims=['s1', 's2'])
    plot_obj.plot_blocking()
    # plot_obj.plot_blocking_per_request()
    # plot_obj.plot_transponders()
    # plot_obj.plot_slots_taken()
    # plot_obj.plot_active_requests()
    # plot_obj.plot_guard_bands()
    # plot_obj.plot_num_segments()
    # plot_obj.plot_dist_cong()
    # plot_obj.plot_bandwidths()


if __name__ == '__main__':
    main()
