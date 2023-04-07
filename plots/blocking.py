import os
import json
import numpy as np
import matplotlib.pyplot as plt  # pylint: disable=import-error

from useful_functions.handle_dirs_files import create_dir


# TODO: Need to change the name of the plot to differentiate the type (another sub-directory is probably the best)
# TODO: Label figures?


class Blocking:
    """
    Creates and saves plot of blocking percentage vs. Erlang.
    """

    def __init__(self):
        self.des_times = None
        self.info_dict = None
        self.network_name = 'USNet'
        self.base_path = f'../data/output/{self.network_name}'
        self.file_info = None

        self.erlang_arr = np.array([])
        self.blocking_arr = np.array([])
        self.trans_arr = np.array([])
        self.dist_arr = np.array([])
        self.cong_arr = np.array([])
        self.band_one = np.array([])
        self.band_two = np.array([])
        self.band_three = np.array([])

        self.plot_dict = dict()
        self.x_axis = [10, 100, 200, 300, 400]

        self.mu = None  # pylint: disable=invalid-name
        self.num_cores = None
        self.spectral_slots = None
        self.lps = True

        self.policy = None
        self.colors = ['#024de3', '#00b300', 'orange', '#6804cc', '#e30220']
        self.line_styles = ['solid', 'dashed', 'dotted', 'dashdot']
        self.markers = {1: 'o', 2: '^', 4: 's', 8: 'x'}
        self.cong_only = False

        self.weighted = False
        self.legend_list = list()

        self.occ_slots = dict()
        self.num_slices = dict()

    def save_show_plot(self, file_name=None):
        """
        Saves and shows the current plot in the constructor.
        """
        # Save by the most recent time for used plotting
        create_dir(f'./output/{self.network_name}/{self.des_times[-1]}/')
        plt.savefig(f'./output/{self.network_name}/{self.des_times[-1]}/{file_name}')
        plt.show()

    def get_file_names(self):
        """
        Obtains all the filenames of the output data for each Erlang value.
        """
        # Default to the last saved directory if one wasn't specified
        if self.des_times is None:
            tmp_list = list()
            for directory in os.listdir(self.base_path):
                tmp_list.append(directory)

            tmp_list.sort()
            latest_date = tmp_list[-1]
            tmp_list = list()

            for sub_dir in os.listdir(f'{self.base_path}/{latest_date}'):
                tmp_list.append(sub_dir)

            tmp_list.sort()
            latest_time = tmp_list[-1]
            self.des_times = f'{latest_date}/{latest_time}'
            create_dir(f'./output/{self.network_name}/{latest_date}/{latest_time}')

        tmp_dict = dict()
        res_dict = dict()
        for curr_time in self.des_times:
            curr_fp = f'{self.base_path}/{curr_time}/'
            res_list = sorted([float(f.split('_')[0]) for f in os.listdir(curr_fp)
                               if os.path.isfile(os.path.join(curr_fp, f))])

            # Then we must have used threading
            if len(res_list) == 0:
                tmp_list = sorted(os.listdir(curr_fp))

                for thread_num in tmp_list:
                    tmp_fp = f'{curr_fp}{thread_num}/'
                    res_list = sorted([float(f.split('_')[0]) for f in os.listdir(tmp_fp)
                                       if os.path.isfile(os.path.join(tmp_fp, f))])
                    tmp_dict[thread_num] = res_list
            else:
                tmp_dict['None'] = res_list

            res_dict[curr_time] = tmp_dict
            tmp_dict = dict()

        return res_dict

    @staticmethod
    def get_weighted_blocking(user_dict):
        """
        Gets the weighted blocking probability.
        """
        block_50 = 0.3 * user_dict['misc_stats']['misc_info']['blocking_obj']['50']
        block_100 = 0.5 * user_dict['misc_stats']['misc_info']['blocking_obj']['100']
        block_400 = 0.2 * user_dict['misc_stats']['misc_info']['blocking_obj']['400']

        return (block_50 + block_100 + block_400) / (user_dict['misc_stats']['num_req'])

    @staticmethod
    def get_blocking(user_dict):
        """
        Gets the vanilla blocking probability (the mean).
        """
        blocking_mean = user_dict['misc_stats']['blocking_mean']
        # Only one iteration occurred, no mean calculated for now
        if blocking_mean is None:
            blocking_mean = user_dict['simulations']['0']

        blocking_mean = float(blocking_mean)
        return blocking_mean

    def plot_blocking(self):
        """
        Plots blocking means for all bandwidths vs. Erlang values.
        """
        plt.figure(figsize=(7, 5), dpi=300)
        if self.weighted and self.cong_only:
            plt.title(f'{self.network_name} Cong. Weighted BP vs. Erlang ({self.policy})')
        elif self.weighted:
            plt.title(f'{self.network_name} Weighted BP vs. Erlang ({self.policy})')
        elif self.cong_only:
            plt.title(f'{self.network_name} 400 Gbps Cong. Only BP vs. Erlang ({self.policy})')
        else:
            plt.title(f'{self.network_name} LS = 8 BP vs. Erlang ({self.policy})')
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
                # marker = self.markers[info_obj['max_lps']]
                plt.plot(info_obj['erlang'], info_obj['blocking'], color=color, linestyle=line_style,
                         markersize=2.3)
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
                # marker = self.markers[obj_2['max_lps']]
                plt.plot(obj_2['erlang'], obj_2['av_trans'], color=color, markersize=2.3)
                # legend_list.append(f"C={obj_2['num_cores']} LS={obj_2['max_lps']}")
                legend_list.append(f"C={obj_2['num_cores']}")

                count += 1
            # count += 1

        plt.legend(legend_list)
        plt.xticks(self.x_axis)
        plt.xlim(self.x_axis[0], self.x_axis[-1])
        plt.ylim(0.9, 3.5)
        self.save_show_plot(file_name='transponders')

    def plot_block_percents(self):
        """
        Plot the reason for blocking as a percentage e.g. due to congestion or distance constraint.
        """
        figure, axis = plt.subplots(2, 2, figsize=(7, 5), dpi=300)  # pylint: disable=unused-variable
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        tmp_lst = [[0, 0], [0, 1], [1, 0], [1, 1]]
        cnt = 0
        sub_cnt = 0

        cong_colors = ['#800000', '#ff1a1a', '#ff6666']
        dist_colors = ['#009900', '#00e600', '#80ff80']

        for curr_time, obj in self.plot_dict.items():
            for thread, obj_2 in obj.items():  # pylint: disable=unused-variable
                axis[tmp_lst[sub_cnt][0], tmp_lst[sub_cnt][1]].grid()
                axis[tmp_lst[sub_cnt][0], tmp_lst[sub_cnt][1]].set_yticks([20, 40, 60, 80, 100])
                axis[tmp_lst[sub_cnt][0], tmp_lst[sub_cnt][1]].plot(obj_2['erlang'], obj_2['cong_block'],
                                                                    color=cong_colors[cnt])
                axis[tmp_lst[sub_cnt][0], tmp_lst[sub_cnt][1]].plot(obj_2['erlang'], obj_2['dist_block'],
                                                                    color=dist_colors[cnt])
                axis[tmp_lst[sub_cnt][0], tmp_lst[sub_cnt][1]].set_xlim(self.x_axis[0], self.x_axis[-1])
                axis[tmp_lst[sub_cnt][0], tmp_lst[sub_cnt][1]].set_ylim(-1, 101)
                axis[tmp_lst[sub_cnt][0], tmp_lst[sub_cnt][1]].set_xticks(self.x_axis)

                if sub_cnt == 0:
                    axis[tmp_lst[sub_cnt][0], tmp_lst[sub_cnt][1]].set_title(
                        f"{self.network_name} 400 Gbps LS = {obj_2['max_lps']} ({self.policy})")
                    axis[tmp_lst[sub_cnt][0], tmp_lst[sub_cnt][1]].legend(['Cong.', 'Dist.'])
                    plt.ylabel('Percent')
                    plt.xlabel('Erlang')
                else:
                    axis[tmp_lst[sub_cnt][0], tmp_lst[sub_cnt][1]].set_title(
                        f"{self.network_name} LS = {obj_2['max_lps']}")

                if sub_cnt == 2:
                    axis[tmp_lst[sub_cnt][0], tmp_lst[sub_cnt][1]].legend(['C=1', 'C=1', 'C=4', 'C=4', 'C=7', 'C=7'])

                sub_cnt += 1
                # Ignore light segment slicing for 16, aka thread 5
                if sub_cnt == 4:
                    break
            sub_cnt = 0
            cnt += 1

        self.save_show_plot(file_name='percents')

    def plot_bandwidths(self):
        """
        Plot the percentage of blocking for each bandwidth individually.
        """
        figure, axis = plt.subplots(2, 2)  # pylint: disable=unused-variable
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        tmp_lst = [[0, 0], [0, 1], [1, 0], [1, 1]]
        cnt = 0

        for thread, obj in self.plot_dict.items():  # pylint: disable=unused-variable
            axis[tmp_lst[cnt][0], tmp_lst[cnt][1]].grid()
            axis[tmp_lst[cnt][0], tmp_lst[cnt][1]].plot(obj['erlang'], obj['50'])
            axis[tmp_lst[cnt][0], tmp_lst[cnt][1]].plot(obj['erlang'], obj['100'])
            axis[tmp_lst[cnt][0], tmp_lst[cnt][1]].plot(obj['erlang'], obj['400'])
            axis[tmp_lst[cnt][0], tmp_lst[cnt][1]].set_xlim(self.x_axis[0], self.x_axis[-1])
            axis[tmp_lst[cnt][0], tmp_lst[cnt][1]].set_ylim(0, 101)
            axis[tmp_lst[cnt][0], tmp_lst[cnt][1]].set_yticks([20, 40, 60, 80, 100])
            axis[tmp_lst[cnt][0], tmp_lst[cnt][1]].set_xticks(self.x_axis)

            if cnt == 0:
                axis[tmp_lst[cnt][0], tmp_lst[cnt][1]].set_title(
                    f"{self.network_name} (Cores, LS = {self.num_cores}, {obj['max_lps']}) {self.policy}")
                axis[tmp_lst[cnt][0], tmp_lst[cnt][1]].legend(['50', '100', '400'])
                plt.ylabel('Percent')
                plt.xlabel('Erlang')
            else:
                axis[tmp_lst[cnt][0], tmp_lst[cnt][1]].set_title(
                    f"{self.network_name} (Cores, LS = {self.num_cores}, {obj['max_lps']})")

            cnt += 1
            # Ignore light segment slicing for 16, aka thread 5
            if cnt == 4:
                break
        self.save_show_plot(file_name='bandwidths')

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

    @staticmethod
    def sorted_simple_dict(d):
        return {k: v for k, v in sorted(d.items())}

    def get_data(self):  # pylint: disable=too-many-statements
        """
        Retrieves all the desired data for plotting.
        """
        for curr_time, curr_dict in self.file_info.items():
            self.plot_dict[curr_time] = dict()
            tmp_dict = dict()

            for thread_num, lst in curr_dict.items():
                self.occ_slots = dict()
                self.num_slices = dict()
                for erlang in lst:
                    if thread_num == 'None':
                        curr_fp = f'{self.base_path}/{curr_time}/'
                    else:
                        curr_fp = f'{self.base_path}/{curr_time}/{thread_num}/'

                    with open(f'{curr_fp}/{erlang}_erlang.json', 'r', encoding='utf-8') as curr_f:
                        curr_dict = json.load(curr_f)

                    self.erlang_arr = np.append(self.erlang_arr, erlang)

                    if not self.weighted:
                        self.blocking_arr = np.append(self.blocking_arr, self.get_blocking(curr_dict))
                    else:
                        self.blocking_arr = np.append(self.blocking_arr, self.get_weighted_blocking(curr_dict))

                    self.trans_arr = np.append(self.trans_arr, curr_dict['misc_stats']['trans_mean'])
                    self.dist_arr = np.append(self.dist_arr, curr_dict['misc_stats']['dist_percent'])
                    self.cong_arr = np.append(self.cong_arr, curr_dict['misc_stats']['cong_percent'])

                    bandwidth_obj = curr_dict['misc_stats']['block_per_bw']
                    total_blocks = bandwidth_obj['50'] + bandwidth_obj['100'] + bandwidth_obj['400']

                    if total_blocks == 0:
                        self.band_one = np.append(self.band_one, 0)
                        self.band_two = np.append(self.band_two, 0)
                        self.band_three = np.append(self.band_three, 0)
                    else:
                        self.band_one = np.append(self.band_one, ((float(bandwidth_obj['50']) / total_blocks) * 100.0))
                        self.band_two = np.append(self.band_two, ((float(bandwidth_obj['100']) / total_blocks) * 100.0))
                        self.band_three = np.append(self.band_three,
                                                    ((float(bandwidth_obj['400']) / total_blocks) * 100.0))

                    self.mu = curr_dict['misc_stats']['hold_time_mean']
                    self.num_cores = curr_dict['misc_stats']['cores_per_link']
                    self.spectral_slots = curr_dict['misc_stats']['spectral_slots']
                    if self.lps:
                        max_lps = curr_dict['misc_stats']['max_slices']
                    else:
                        max_lps = None

                    if erlang == 10 or erlang == 100 or erlang == 700:
                        self.occ_slots[erlang] = dict()
                        self.num_slices[erlang] = list()

                        for req_id, ss_obj in curr_dict['misc_stats']['slot_slice_dict'].items():
                            self.num_slices[erlang].append(ss_obj['num_slices'])

                        self.occ_slots[erlang]['req_ids'] = [curr_id for curr_id in range(0, 11000, 1000)]
                        self.occ_slots[erlang]['occ_slots'] = list()
                        for req_id in self.occ_slots[erlang]['req_ids']:
                            if req_id == 0:
                                req_id = 1
                            self.occ_slots[erlang]['occ_slots'].append(
                                curr_dict['misc_stats']['slot_slice_dict'][f'{req_id}']['occ_slots'])

                    # Only plot up to Erlang 400, beyond that is considered not important for the time being
                    if erlang == self.x_axis[-1]:
                        break

                thread_num = int(thread_num[1:])
                tmp_dict[thread_num] = dict()

                tmp_dict[thread_num]['erlang'] = self.erlang_arr
                tmp_dict[thread_num]['blocking'] = self.blocking_arr
                tmp_dict[thread_num]['av_trans'] = self.trans_arr
                tmp_dict[thread_num]['max_lps'] = max_lps
                tmp_dict[thread_num]['cong_block'] = self.cong_arr
                tmp_dict[thread_num]['dist_block'] = self.dist_arr
                tmp_dict[thread_num]['num_cores'] = curr_dict['misc_stats']['cores_per_link']
                tmp_dict[thread_num]['50'] = self.band_one
                tmp_dict[thread_num]['100'] = self.band_two
                tmp_dict[thread_num]['400'] = self.band_three
                tmp_dict[thread_num]['occ_slots'] = self.occ_slots
                tmp_dict[thread_num]['num_slices'] = self.num_slices

                self.erlang_arr = np.array([])
                self.blocking_arr = np.array([])
                self.trans_arr = np.array([])
                self.dist_arr = np.array([])
                self.cong_arr = np.array([])
                self.band_one = np.array([])
                self.band_two = np.array([])
                self.band_three = np.array([])
                self.erlang_arr = np.array([])

            self.plot_dict[curr_time] = tmp_dict
        # Sort thread numbers for each time
        self.plot_dict = {k: self.sorted_simple_dict(v) for k, v in sorted(self.plot_dict.items())}


def main():
    """
    Controls the plotting object.
    :return: None
    """
    # TODO: Clean this up, make it more efficient
    blocking_obj = Blocking()

    # blocking_obj.des_times = ['0329/11:50:35', '0323/09:22:02', '0323/09:22:04']
    blocking_obj.des_times = ['0407/15:53:41']
    blocking_obj.policy = 'First Fit'
    blocking_obj.weighted = False
    blocking_obj.cong_only = False
    blocking_obj.file_info = blocking_obj.get_file_names()
    blocking_obj.get_data()
    # blocking_obj.plot_slots_taken()
    # blocking_obj.plot_num_slices()
    blocking_obj.plot_blocking()
    blocking_obj.plot_transponders()
    # blocking_obj.plot_block_percents()


if __name__ == '__main__':
    main()
