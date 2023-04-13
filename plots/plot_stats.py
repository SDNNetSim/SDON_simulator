# Standard library imports
import os
import json

# Third-party imports
import matplotlib.pyplot as plt

# Local application imports
from useful_functions.handle_dirs_files import create_dir


class PlotStats:
    """
    A class for computing and plotting statistical analysis for simulations.
    """

    def __init__(self, net_name: str, data_dir: str = None, latest_date: str = None, latest_time: str = None):
        """
        Initializes the PlotStats class.
        """
        # Information to retrieve desired data
        self.net_name = net_name
        self.data_dir = data_dir
        self.latest_date = latest_date
        self.latest_time = latest_time
        self.file_info = self.get_file_info()

        # The final dictionary containing information for all plots
        self.plot_dict = {}
        # Miscellaneous customizations visually
        self.colors = ['#024de3', '#00b300', 'orange', '#6804cc', '#e30220']
        self.line_styles = ['solid', 'dashed', 'dotted', 'dashdot']
        self.markers = {1: 'o', 2: '^', 4: 's', 8: 'x'}
        self.x_ticks = [10, 100, 200, 300, 400]

        self.get_data()

    def get_file_info(self):
        """
        Obtains all the filenames of the output data from the simulations.
        """
        # Default to the latest time if a data directory wasn't specified
        if self.data_dir is None:
            self.data_dir = f'../data/output/{self.net_name}/'
            dir_list = os.listdir(self.data_dir)
            self.latest_date = sorted(dir_list)[-1]
            dir_list = os.listdir(os.path.join(self.data_dir, self.latest_date))
            self.latest_time = sorted(dir_list)[-1]
            self.data_dir = os.path.join(self.data_dir, f'{self.latest_date}/{self.latest_time}')

        files_dict = {}
        for thread in os.listdir(self.data_dir):
            curr_fp = os.path.join(self.data_dir, thread)
            files_dict[thread] = list()
            for erlang_file in os.listdir(curr_fp):
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
                'max_slices': None,
                'num_slices': {},
                'taken_slots': {},
            }
            for erlang in erlang_values:
                curr_fp = os.path.join(self.data_dir, f'{thread}/{erlang}_erlang.json')
                with open(curr_fp, 'r', encoding='utf-8') as file_obj:
                    erlang_dict = json.load(file_obj)

                erlang = int(erlang.split('.')[0])
                self.plot_dict[thread]['erlang_vals'].append(erlang)

                # Only one iteration occurred, a mean was not calculated
                if erlang_dict['misc_stats']['blocking_mean'] is None:
                    self.plot_dict[thread]['blocking_vals'].append(erlang_dict['misc_stats']['block_per_sim'][0])
                else:
                    self.plot_dict[thread]['blocking_vals'].append(erlang_dict['misc_stats']['blocking_mean'])

                self.plot_dict[thread]['average_transponders'].append(erlang_dict['misc_stats']['trans_mean'])
                self.plot_dict[thread]['distance_block'].append(erlang_dict['misc_stats']['dist_percent'])
                self.plot_dict[thread]['cong_block'].append(erlang_dict['misc_stats']['cong_percent'])

                self.plot_dict[thread]['hold_time_mean'] = erlang_dict['misc_stats']['hold_time_mean']
                self.plot_dict[thread]['cores_per_link'] = erlang_dict['misc_stats']['cores_per_link']
                self.plot_dict[thread]['spectral_slots'] = erlang_dict['misc_stats']['spectral_slots']

                self.plot_dict[thread]['max_slices'] = erlang_dict['misc_stats']['max_slices']

                if erlang in (10, 100, 700):
                    self.plot_dict[thread]['taken_slots'][erlang] = dict()
                    self.plot_dict[thread]['num_slices'][erlang] = list()

                    for request_number, request_info in erlang_dict['misc_stats']['slot_slice_dict'].items():
                        request_number = int(request_number)
                        if request_number % 1000 == 0 or request_number == 1:
                            self.plot_dict[thread]['taken_slots'][erlang][request_number] = request_info['occ_slots'] / \
                                                                                            erlang_dict['misc_stats'][
                                                                                                'cores_per_link']

                        self.plot_dict[thread]['num_slices'][erlang].append(request_info['num_slices'])

    def _save_plot(self, file_name):
        """
        Save a Matplotlib plot to a file with the specified name.

        :param file_name: A string representing the name of the file to save the plot as
        :type file_name: str
        """
        file_path = f'./output/{self.net_name}/{self.latest_date}/{self.latest_time}/'
        create_dir(file_path)
        plt.savefig(f'{file_path}/{file_name}.png')

    def _setup_plot(self, title, y_label, x_label, grid=True, x_ticks=True):
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

        :param x_ticks: Determines if we'd like to plot the default for x_ticks, which are Erlang values
        :type x_ticks: bool
        """
        plt.figure(figsize=(7, 5), dpi=300)
        plt.title(title)
        plt.ylabel(y_label)
        plt.xlabel(x_label)

        if x_ticks:
            plt.xticks(self.x_ticks)
            plt.xlim(self.x_ticks[0], self.x_ticks[-1])

        if grid:
            plt.grid()

    def plot_blocking(self):
        """
        Plots the blocking probability for each Erlang value.
        """
        self._setup_plot(f'{self.net_name} BP vs. Erlang', 'Blocking Probability', 'Erlang')

        plt.ylim(10 ** -5, 1)
        plt.yscale('log')

        style_count = 0
        legend_list = list()
        for thread, thread_obj in self.plot_dict.items():  # pylint: disable=unused-variable
            color = self.colors[style_count]
            line_style = self.line_styles[style_count]
            marker = self.markers[thread_obj['max_slices']]

            plt.plot(thread_obj['erlang_vals'], thread_obj['blocking_vals'], color=color, linestyle=line_style,
                     marker=marker, markersize=2.3)
            legend_list.append(f"C ={thread_obj['cores_per_link']} LS ={thread_obj['max_slices']}")

            style_count += 1

        plt.yticks([10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1])
        plt.legend(legend_list)

        self._save_plot(file_name='blocking')
        plt.show()

    def plot_transponders(self):
        """
        Plots the average number of transponders used for each Erlang value.
        """
        self._setup_plot(f'{self.net_name} Transponders vs. Erlang', 'Transponders', 'Erlang')
        plt.ylim(0.9, 2)

        legend_list = list()
        style_count = 0
        for thread, thread_obj in self.plot_dict.items():  # pylint: disable=unused-variable
            color = self.colors[style_count]

            plt.plot(thread_obj['erlang_vals'], thread_obj['average_transponders'], color=color)
            legend_list.append(f"C={thread_obj['cores_per_link']} LS={thread_obj['max_slices']}")

            style_count += 1

        plt.legend(legend_list)
        self._save_plot(file_name='transponders')
        plt.show()

    def plot_slots_taken(self):
        """
        Plots the number of slots taken in the entire network for certain request snapshots.
        """
        self._setup_plot(f'{self.net_name} Request Snapshot vs. Slots Occupied', 'Slots Occupied', 'Request Number',
                         x_ticks=False)

        legend_list = list()
        style_count = 0
        for _, thread_obj in self.plot_dict.items():
            color = self.colors[style_count]

            for erlang in thread_obj['taken_slots']:
                marker = self.markers[thread_obj['max_slices']]

                request_numbers = thread_obj['taken_slots'][erlang].keys()
                slots_occupied = thread_obj['taken_slots'][erlang].values()
                plt.plot(request_numbers, slots_occupied, color=color, marker=marker, markersize=2.3)

                legend_list.append(f"E={erlang} LS={thread_obj['max_slices']}")

            style_count += 1

        plt.legend(legend_list, loc='upper left')
        plt.xlim(0, 10000)
        plt.ylim(0, 4100)
        self._save_plot(file_name='slots_occupied')
        plt.show()

    def plot_num_slices(self):
        """
        Plots the number of times all requests have been sliced.
        """
        self._setup_plot(f'{self.net_name} Number of Slices vs. Occurrences', 'Occurrences', 'Number of Slices',
                         x_ticks=False, grid=False)

        erlang_colors = ['#0000b3', '#3333ff', '#9999ff', '#b30000', '#ff3333', '#ff9999', '#00b33c', '#00ff55',
                         '#99ffbb', '#b36b00', '#ff9900', '#ffb84d']

        hist_list = list()
        legend_list = list()
        for _, thread_obj in self.plot_dict.items():
            for erlang, slices_lst in thread_obj['num_slices'].items():
                hist_list.append(slices_lst)
                legend_list.append(f"E={int(erlang)} LS={thread_obj['max_slices']}")

        plt.hist(hist_list, stacked=False, histtype='bar', edgecolor='black', rwidth=1, color=erlang_colors)

        plt.ylim(0, 10000)
        plt.xlim(0, 8)
        plt.legend(legend_list, loc='upper right')
        self._save_plot(file_name='num_slices')
        plt.show()


def main():
    """
    Controls this script.
    """
    plot_obj = PlotStats(net_name='USNet')
    # plot_obj.plot_blocking()
    # plot_obj.plot_transponders()
    # plot_obj.plot_slots_taken()
    plot_obj.plot_num_slices()


if __name__ == '__main__':
    main()
