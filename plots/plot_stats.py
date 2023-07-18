# Standard library imports
import os
import json

# Third-party imports
import matplotlib.pyplot as plt

# Local application imports
from useful_functions.handle_dirs_files import create_dir


# TODO: Bring back congestion vs. distance blocking


class PlotStats:
    """
    A class for computing and plotting statistical analysis for simulations.
    """

    def __init__(self, net_name: str, data_dir: str = None, latest_date: str = None, latest_time: str = None,
                 plot_threads: list = None):
        """
        Initializes the PlotStats class.
        """
        # Information to retrieve desired data
        self.net_name = net_name
        self.data_dir = data_dir
        self.latest_date = latest_date
        self.latest_time = latest_time
        # Desired threads to be plotted
        self.plot_threads = plot_threads
        self.file_info = self.get_file_info()

        # The final dictionary containing information for all plots
        self.plot_dict = {}
        self.num_cores = None
        # Miscellaneous customizations visually
        self.colors = ['#024de3', '#00b300', 'orange', '#6804cc', '#e30220']
        self.line_styles = ['solid', 'dashed', 'dotted', 'dashdot']
        self.markers = ['o', '^', 's', 'x']
        self.x_ticks = [10, 100, 200, 300, 400, 500, 600, 700]

        self.get_data()

    def get_file_info(self):
        """
        Obtains all the filenames of the output data from the simulations.
        """
        # Default to the latest time if a data directory wasn't specified
        if self.latest_date is None or self.latest_time is None:
            self.data_dir = f'../data/output/{self.net_name}/'
            dir_list = os.listdir(self.data_dir)
            self.latest_date = sorted(dir_list)[-1]
            dir_list = os.listdir(os.path.join(self.data_dir, self.latest_date))
            self.latest_time = sorted(dir_list)[-1]
            self.data_dir = os.path.join(self.data_dir, f'{self.latest_date}/{self.latest_time}')

        files_dict = {}
        self.data_dir = f'../data/output/{self.net_name}/{self.latest_date}/{self.latest_time}'

        # Sort by thread number
        dirs = os.listdir(self.data_dir)
        sorted_dirs = sorted(dirs, key=lambda x: int(x[1:]))
        for thread in sorted_dirs:
            if thread not in self.plot_threads:
                continue
            curr_fp = os.path.join(self.data_dir, thread)
            files_dict[thread] = list()

            files = os.listdir(curr_fp)
            sorted_files = sorted(files, key=lambda x: float(x.split('_')[0]))
            for erlang_file in sorted_files:
                files_dict[thread].append(erlang_file.split('_')[0])

        return files_dict

    def get_data(self):
        """
        Structures all data to be plotted.
        """
        for thread, erlang_values in self.file_info.items():
            self.plot_dict[thread] = {
                'erlang_vals': [],
                'blocking_vals': [],
                'average_transponders': [],
                'distance_block': [],
                'cong_block': [],
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
            for erlang in erlang_values:
                curr_fp = os.path.join(self.data_dir, f'{thread}/{erlang}_erlang.json')
                with open(curr_fp, 'r', encoding='utf-8') as file_obj:
                    erlang_dict = json.load(file_obj)

                erlang = int(erlang.split('.')[0])
                self.plot_dict[thread]['erlang_vals'].append(erlang)

                # Only one iteration occurred, a mean was not calculated
                if erlang_dict['misc_stats']['blocking_mean'] is None:
                    self.plot_dict[thread]['blocking_vals'].append(erlang_dict['block_per_sim']["0"])
                else:
                    self.plot_dict[thread]['blocking_vals'].append(erlang_dict['misc_stats']['blocking_mean'])

                self.plot_dict[thread]['average_transponders'].append(erlang_dict['misc_stats']['trans_mean'])
                self.plot_dict[thread]['distance_block'].append(erlang_dict['misc_stats']['dist_percent'])
                self.plot_dict[thread]['cong_block'].append(erlang_dict['misc_stats']['cong_percent'])

                self.plot_dict[thread]['hold_time_mean'] = erlang_dict['misc_stats']['hold_time_mean']
                self.plot_dict[thread]['cores_per_link'] = erlang_dict['misc_stats']['cores_per_link']
                self.plot_dict[thread]['spectral_slots'] = erlang_dict['misc_stats']['spectral_slots']

                self.plot_dict[thread]['max_segments'] = erlang_dict['misc_stats']['max_segments']
                self.num_cores = erlang_dict['misc_stats']['cores_per_link']

                if erlang in (10, 100, 400):
                    self.plot_dict[thread]['taken_slots'][erlang] = dict()
                    self.plot_dict[thread]['block_per_req'][erlang] = dict()
                    self.plot_dict[thread]['active_requests'][erlang] = dict()
                    self.plot_dict[thread]['guard_bands'][erlang] = dict()
                    self.plot_dict[thread]['num_segments'][erlang] = list()

                    for request_number, request_info in erlang_dict['misc_stats']['request_snapshots'].items():
                        request_number = int(request_number)
                        # Plot every 'x' amount of request snapshots
                        if request_number % 1 == 0 or request_number == 1:
                            self.plot_dict[thread]['taken_slots'][erlang][request_number] = request_info['occ_slots'] / \
                                                                                            erlang_dict['misc_stats'][
                                                                                                'cores_per_link']
                            self.plot_dict[thread]['block_per_req'][erlang][request_number] = request_info[
                                'blocking_prob']
                            self.plot_dict[thread]['active_requests'][erlang][request_number] = request_info[
                                                                                                    'active_requests'] / \
                                                                                                erlang_dict[
                                                                                                    'misc_stats'][
                                                                                                    'cores_per_link']

                            self.plot_dict[thread]['guard_bands'][erlang][request_number] = request_info[
                                                                                                'guard_bands'] / \
                                                                                            erlang_dict['misc_stats'][
                                                                                                'cores_per_link']

                        self.plot_dict[thread]['num_segments'][erlang].append(request_info['num_segments'])

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
        file_path = f'./output/{self.net_name}/{self.latest_date}/{self.latest_time}'
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

    def plot_blocking(self):
        """
        Plots the blocking probability for each Erlang value.
        """
        self._setup_plot(f'{self.net_name} BP vs. Erlang (C={self.num_cores})', 'Blocking Probability', 'Erlang')

        style_count = 0
        legend_list = list()
        for _, thread_obj in self.plot_dict.items():
            color = self.colors[style_count]
            line_style = self.line_styles[style_count]
            marker = self.markers[style_count]

            plt.plot(thread_obj['erlang_vals'], thread_obj['blocking_vals'], color=color, linestyle=line_style,
                     marker=marker, markersize=2.3)
            legend_list.append(f"C ={thread_obj['cores_per_link']} LS ={thread_obj['max_segments']}")

            style_count += 1

        plt.yticks([10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1])
        plt.legend(legend_list)

        self._save_plot(file_name='blocking')
        plt.show()

    def plot_blocking_per_request(self):
        """
        Plots the blocking probability, but for each request at that point in time.
        """
        self._setup_plot(f'{self.net_name} Request Snapshot vs. Blocking Probability (C={self.num_cores})',
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
        self._setup_plot(f'{self.net_name} Transponders vs. Erlang (C={self.num_cores})', 'Transponders', 'Erlang',
                         y_ticks=False)
        plt.ylim(0.9, 1.6)

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
        self._setup_plot(f'{self.net_name} Request Snapshot vs. Slots Occupied (C={self.num_cores})', 'Slots Occupied',
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

    def plot_num_segments(self):
        """
        Plots the number of segments each request has been sliced into.
        """
        self._setup_plot(f'{self.net_name} Number of Segments vs. Occurrences (C={self.num_cores})', 'Occurrences',
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

        plt.ylim(0, 10000)
        plt.xlim(0, 8)
        plt.legend(legend_list, loc='upper right')
        self._save_plot(file_name='num_segments')
        plt.show()

    def plot_active_requests(self):
        """
        Plots the amount of active requests in the network at specific intervals.
        """
        self._setup_plot(f'{self.net_name} Request Snapshot vs. Active Requests (C={self.num_cores})',
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
        self._setup_plot(f'{self.net_name} Request Snapshot vs. Guard Bands (C={self.num_cores})', 'Guard Bands',
                         'Request Number',
                         y_ticks=False, x_ticks=False)

        legend_list = list()
        style_count = 0
        marker_count = 1
        for _, thread_obj in self.plot_dict.items():
            color = self.colors[style_count]
            for erlang in thread_obj['guard_bands']:
                lst = list(thread_obj['guard_bands'][erlang].values())
                request_numbers, guard_bands = self.running_average(lst=lst, interval=500)

                marker = self.markers[marker_count]
                plt.plot(request_numbers, guard_bands, color=color, marker=marker, markersize=2.3)

                legend_list.append(f"E={erlang} LS={thread_obj['max_segments']}")
                marker_count += 1

            marker_count = 1
            style_count += 1

        plt.legend(legend_list, loc='upper left')
        plt.xlim(1000, 10000)
        plt.ylim(0, 1200)
        self._save_plot(file_name='guard_bands')
        plt.show()


def main():
    """
    Controls this script.
    """
    plot_obj = PlotStats(net_name='USNet', latest_date='0718', latest_time='14:53:41',
                         plot_threads=['t1'])
    plot_obj.plot_blocking()
    # plot_obj.plot_blocking_per_request()
    # plot_obj.plot_transponders()
    # plot_obj.plot_slots_taken()
    # plot_obj.plot_active_requests()
    # plot_obj.plot_guard_bands()
    # plot_obj.plot_num_segments()


if __name__ == '__main__':
    main()
