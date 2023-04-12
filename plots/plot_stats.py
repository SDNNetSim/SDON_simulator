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

    def __init__(self, net_name: str, data_dir: str = None):
        """
        Initializes the PlotStats class.
        """
        # Information to retrieve desired data
        self.net_name = net_name
        self.data_dir = data_dir
        self.file_info = self.get_file_info()

        # The final dictionary containing information for all plots
        self.plot_dict = {}
        # Miscellaneous customizations visually
        self.colors = ['#024de3', '#00b300', 'orange', '#6804cc', '#e30220']
        self.line_styles = ['solid', 'dashed', 'dotted', 'dashdot']
        self.markers = {1: 'o', 2: '^', 4: 's', 8: 'x'}

        self.get_data()

    def get_file_info(self):
        """
        Obtains all the filenames of the output data from the simulations.
        """
        # Default to the latest time if a data directory wasn't specified
        if self.data_dir is None:
            self.data_dir = f'../data/output/{self.net_name}/'
            dir_list = os.listdir(self.data_dir)
            latest_date = sorted(dir_list)[-1]
            dir_list = os.listdir(os.path.join(self.data_dir, latest_date))
            latest_time = sorted(dir_list)[-1]
            self.data_dir = os.path.join(self.data_dir, f'{latest_date}/{latest_time}')

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
                'average_transistors': [],
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

                self.plot_dict[thread]['average_transistors'].append(erlang_dict['misc_stats']['trans_mean'])
                self.plot_dict[thread]['distance_block'].append(erlang_dict['misc_stats']['dist_percent'])
                self.plot_dict[thread]['cong_block'].append(erlang_dict['misc_stats']['cong_percent'])

                self.plot_dict[thread]['hold_time_mean'] = erlang_dict['misc_stats']['hold_time_mean']
                self.plot_dict[thread]['cores_per_link'] = erlang_dict['misc_stats']['cores_per_link']
                self.plot_dict[thread]['spectral_slots'] = erlang_dict['misc_stats']['spectral_slots']

                self.plot_dict[thread]['max_slices'] = erlang_dict['misc_stats']['max_slices']

                if erlang == 10 or erlang == 100 or erlang == 700:
                    self.plot_dict[thread]['taken_slots'][erlang] = dict()
                    self.plot_dict[thread]['num_slices'][erlang] = list()

                    for request_number, request_info in erlang_dict['misc_stats']['slot_slice_dict'].items():
                        if int(request_number) % 1000 == 0 or int(request_number) == 1:
                            self.plot_dict[thread]['taken_slots'][erlang] = request_info['occ_slots']

                        self.plot_dict[thread]['num_slices'][erlang].append(request_info['num_slices'])

    def _save_plot(self):
        pass

    def _show_plot(self):
        pass

    def _setup_plots(self):
        pass

    def plot_blocking(self):
        plt.figure(figsize=(7, 5), dpi=300)
        plt.title(f'{self.net_name} LS = 8 BP vs. Erlang ({self.policy})')
        plt.ylabel('Blocking Probability')
        plt.ylim(10 ** -5, 1)
        plt.yscale('log')

        plt.grid()
        plt.xlabel('Erlang')

        count = 0
        for curr_time, thread_obj in self.plot_dict.items():
            # color = self.colors[count]
            line_style = self.line_styles[count]
            for thread, info_obj in thread_obj.items():  # pylint: disable=unused-variable
                # LS = 16 considered irrelevant for the time being
                color = self.colors[count]
                # if info_obj['max_lps'] != 8 and info_obj['max_lps'] != 8:
                #     continue
                # if info_obj['max_lps'] != 8:
                #     continue
                marker = self.markers[info_obj['max_lps']]
                plt.plot(info_obj['erlang'], info_obj['blocking'], color=color, linestyle=line_style,
                         marker=marker, markersize=2.3)
                # plt.plot(info_obj['erlang'], info_obj['blocking'], color=color,
                #          markersize=2.3)
                self.legend_list.append(f"C ={info_obj['num_cores']} LS ={info_obj['max_lps']}")
                # self.legend_list.append(f"C ={info_obj['num_cores']}")

                count += 1
            # count += 1

        plt.yticks([10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1])
        plt.xticks(self.x_axis)
        plt.xlim(self.x_axis[0], self.x_axis[-1])
        plt.legend(self.legend_list)

        self.save_show_plot(file_name='blocking')

    def plot_transponders(self):
        """
        Plots the average number of transponders
        """
        plt.figure(figsize=(7, 5), dpi=300)
        plt.title(f'{self.network_name} LS = 8 Transponders vs. Erlang ({self.policy})')
        plt.ylabel('Transponders per Request')

        plt.grid()
        plt.xlabel('Erlang')

        legend_list = list()
        count = 0
        for curr_time, obj in self.plot_dict.items():
            for thread, obj_2 in obj.items():  # pylint: disable=unused-variable
                color = self.colors[count]
                # if obj_2['max_lps'] != 8:
                #     continue
                marker = self.markers[obj_2['max_lps']]
                plt.plot(obj_2['erlang'], obj_2['av_trans'], color=color, markersize=2.3)
                # legend_list.append(f"C={obj_2['num_cores']} LS={obj_2['max_lps']}")
                legend_list.append(f"C={obj_2['num_cores']}")

                count += 1
            # count += 1

        plt.legend(legend_list)
        plt.xticks(self.x_axis)
        plt.xlim(self.x_axis[0], self.x_axis[-1])
        plt.ylim(0.9, 2)
        self.save_show_plot(file_name='transponders')

    def plot_slots_taken(self):
        plt.figure(figsize=(7, 5), dpi=300)
        # plt.title(f'{self.network_name} Request Snapshot vs. Slots Occupied (C = {self.num_cores})')
        plt.title(f'{self.network_name} Unlimited Slicing Request Snapshot vs. Slots Occupied Norm')
        plt.ylabel('Slots Occupied')

        plt.grid()
        plt.xlabel('Request Number')

        legend_list = list()
        marker_lst = ['o', '^', 's', 'x']
        count = 0
        marker_count = 0
        for curr_time, obj in self.plot_dict.items():
            for thread, obj_2 in obj.items():  # pylint: disable=unused-variable
                for erlang, obj_3 in obj_2['occ_slots'].items():
                    # if erlang == 700 and obj_2['max_lps'] == 8:
                    #     continue
                    color = self.colors[count]
                    marker = marker_lst[marker_count]

                    # print(self.num_cores)
                    # obj_3['occ_slots'] = [val / self.num_cores for val in obj_3['occ_slots']]
                    obj_3['occ_slots'] = [val / obj_2['num_cores'] for val in obj_3['occ_slots']]
                    plt.plot(obj_3['req_ids'], obj_3['occ_slots'], color=color, marker=marker, markersize=2.3)
                    # legend_list.append(f"E={int(erlang)} LS={obj_2['max_lps']}")
                    legend_list.append(f"E={int(erlang)} C={obj_2['num_cores']}")
                    # count += 1
                    marker_count += 1
                    if marker_count == 3:
                        marker_count = 0
                count += 1

        plt.legend(legend_list, loc='upper left')
        plt.xlim(0, 10000)
        plt.ylim(0, 4100)
        self.save_show_plot(file_name='slots_occupied')

    def plot_num_slices(self):
        plt.figure(figsize=(7, 5), dpi=300)
        # plt.title(f'{self.network_name} Number of Slices vs. Occurrences (C = {self.num_cores})')
        plt.title(f'{self.network_name} Number of Slices vs. Occurrences (Unlimited)')
        plt.ylabel('Occurrences')

        plt.xlabel('Number of Slices')

        slice_colors = ['#0000b3', '#3333ff', '#9999ff', '#b30000', '#ff3333', '#ff9999', '#00b33c', '#00ff55',
                        '#99ffbb', '#b36b00', '#ff9900', '#ffb84d']

        count = 0
        res_list = list()
        legend_list = list()
        for curr_time, obj in self.plot_dict.items():
            for thread, obj_2 in obj.items():
                if obj_2['max_lps'] == 1:
                    continue
                for erlang, slice_lst in obj_2['num_slices'].items():
                    res_list.append(slice_lst)
                    legend_list.append(f"E={int(erlang)} C={obj_2['num_cores']}")
                    count += 1
                # count += 1

        plt.hist(res_list,
                 stacked=False, histtype='bar', edgecolor='black', rwidth=1, color=slice_colors)

        plt.ylim(0, 10000)
        plt.xlim(0, 8)
        plt.legend(legend_list, loc='upper right')
        self.save_show_plot(file_name='num_slices')


def main():
    test_obj = PlotStats(net_name='USNet')
    test_obj.plot_blocking()


if __name__ == '__main__':
    main()
