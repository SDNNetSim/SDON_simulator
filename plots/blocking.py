import os
import json
import numpy as np
import matplotlib.pyplot as plt  # pylint: disable=import-error

from useful_functions.handle_dirs_files import create_dir


# TODO: Need to change the name of the plot to differentiate the type (another sub-directory is probably the best)


class Blocking:
    """
    Creates and saves plot of blocking percentage vs. Erlang.
    """

    def __init__(self):
        # Change these variables for the desired plot you'd like
        # self.des_time = '0228/15:20:41'
        self.des_time = None
        self.network_name = 'USNet'
        self.base_path = f'../data/output/{self.network_name}'
        self.file_info = self.get_file_names()

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

    def save_show_plot(self, file_name=None):
        """
        Saves and shows the current plot in the constructor.
        """
        plt.savefig(f'./output/{self.network_name}/{self.des_time}/{file_name}')
        plt.show()

    def get_file_names(self):
        """
        Obtains all the filenames of the output data for each Erlang value.
        """
        res_dict = dict()

        # Default to the last saved directory if one wasn't specified
        if self.des_time is None:
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
            self.des_time = f'{latest_date}/{latest_time}'
            create_dir(f'./output/{self.network_name}/{latest_date}/{latest_time}')

        curr_fp = f'{self.base_path}/{self.des_time}/'
        res_list = sorted([float(f.split('_')[0]) for f in os.listdir(curr_fp)
                           if os.path.isfile(os.path.join(curr_fp, f))])

        # Then we must have used threading
        if len(res_list) == 0:
            tmp_list = sorted(os.listdir(curr_fp))

            for thread_num in tmp_list:
                tmp_fp = f'{curr_fp}{thread_num}/'
                res_list = sorted([float(f.split('_')[0]) for f in os.listdir(tmp_fp)
                                   if os.path.isfile(os.path.join(tmp_fp, f))])
                res_dict[thread_num] = res_list
        else:
            res_dict['None'] = res_list

        return res_dict

    @staticmethod
    def get_weighted_blocking(user_dict):
        """
        Gets the weighted blocking probability.
        """
        block_50 = 50 * user_dict['stats']['misc_info']['blocking_obj']['50']
        block_100 = 100 * user_dict['stats']['misc_info']['blocking_obj']['100']
        block_400 = 400 * user_dict['stats']['misc_info']['blocking_obj']['400']

        total_50 = 50 * (0.3 * user_dict['stats']['num_req'])
        total_100 = 100 * (0.5 * user_dict['stats']['num_req'])
        total_400 = 400 * (0.2 * user_dict['stats']['num_req'])

        return (block_50 + block_100 + block_400) / (total_50 + total_100 + total_400)

    @staticmethod
    def get_blocking(user_dict):
        """
        Gets the vanilla blocking probability (the mean).
        """
        blocking_mean = user_dict['stats']['mean']
        # Only one iteration occurred, no mean calculated for now
        if blocking_mean is None:
            blocking_mean = user_dict['simulations']['0']

        blocking_mean = float(blocking_mean)
        return blocking_mean

    def plot_blocking(self):
        """
        Plots blocking means for all bandwidths vs. Erlang values.
        """
        plt.title(f'{self.network_name} BP vs. Erlang (Cores = {self.num_cores})')
        plt.ylabel('Blocking Probability')
        plt.yscale('log')

        plt.grid()
        plt.xlabel('Erlang')

        legend_list = list()

        for thread, obj in self.plot_dict.items():  # pylint: disable=unused-variable
            plt.plot(obj['erlang'], obj['blocking'])
            legend_list.append(f"LS = {obj['max_lps']}")

        plt.yticks([10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1])
        plt.legend(legend_list)
        plt.xticks(self.x_axis)
        plt.xlim(10, 400)

        self.save_show_plot(file_name='blocking')

    def plot_transponders(self):
        """
        Plots the average number of transponders
        """
        plt.title(f'{self.network_name} Transponders vs. Erlang (Cores = {self.num_cores})')
        plt.ylabel('Transponders per Request')

        plt.grid()
        plt.xlabel('Erlang')

        legend_list = list()
        for thread, obj in self.plot_dict.items():  # pylint: disable=unused-variable
            plt.plot(obj['erlang'], obj['av_trans'])
            legend_list.append(f"LS = {obj['max_lps']}")

        plt.legend(legend_list)
        plt.xticks(self.x_axis)
        plt.xlim(10, 400)
        self.save_show_plot(file_name='transponders')

    def plot_block_percents(self):
        """
        Plot the reason for blocking as a percentage e.g. due to congestion or distance constraint.
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
            axis[tmp_lst[cnt][0], tmp_lst[cnt][1]].set_yticks([20, 40, 60, 80, 100])
            axis[tmp_lst[cnt][0], tmp_lst[cnt][1]].plot(obj['erlang'], obj['cong_block'])
            axis[tmp_lst[cnt][0], tmp_lst[cnt][1]].plot(obj['erlang'], obj['dist_block'])
            axis[tmp_lst[cnt][0], tmp_lst[cnt][1]].set_xlim(10, 400)
            axis[tmp_lst[cnt][0], tmp_lst[cnt][1]].set_ylim(-1, 101)
            axis[tmp_lst[cnt][0], tmp_lst[cnt][1]].set_xticks(self.x_axis)
            axis[tmp_lst[cnt][0], tmp_lst[cnt][1]].set_title(
                f"{self.network_name} (Cores, LS = {self.num_cores}, {obj['max_lps']})")

            if cnt == 0:
                axis[tmp_lst[cnt][0], tmp_lst[cnt][1]].legend(['Congestion', 'Distance'])
                plt.ylabel('Percent')
                plt.xlabel('Erlang')

            cnt += 1
            # Ignore light segment slicing for 16, aka thread 5
            if cnt == 4:
                break

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
            axis[tmp_lst[cnt][0], tmp_lst[cnt][1]].set_xlim(10, 400)
            axis[tmp_lst[cnt][0], tmp_lst[cnt][1]].set_ylim(0, 101)
            axis[tmp_lst[cnt][0], tmp_lst[cnt][1]].set_yticks([20, 40, 60, 80, 100])
            axis[tmp_lst[cnt][0], tmp_lst[cnt][1]].set_xticks(self.x_axis)
            axis[tmp_lst[cnt][0], tmp_lst[cnt][1]].set_title(
                f"{self.network_name} (Cores, LS = {self.num_cores}, {obj['max_lps']})")

            if cnt == 0:
                axis[tmp_lst[cnt][0], tmp_lst[cnt][1]].legend(['50', '100', '400'])
                plt.ylabel('Percent')
                plt.xlabel('Erlang')

            cnt += 1
            # Ignore light segment slicing for 16, aka thread 5
            if cnt == 4:
                break
        self.save_show_plot(file_name='bandwidths')

    def get_data(self):  # pylint: disable=too-many-statements
        """
        Retrieves all the desired data for plotting.
        """
        for thread_num, lst in self.file_info.items():
            for erlang in lst:
                if thread_num == 'None':
                    curr_fp = f'{self.base_path}/{self.des_time}/'
                else:
                    curr_fp = f'{self.base_path}/{self.des_time}/{thread_num}/'

                with open(f'{curr_fp}/{erlang}_erlang.json', 'r', encoding='utf-8') as curr_f:
                    curr_dict = json.load(curr_f)

                self.erlang_arr = np.append(self.erlang_arr, erlang)
                self.blocking_arr = np.append(self.blocking_arr, self.get_blocking(curr_dict))
                # self.blocking_arr = np.append(self.blocking_arr, self.get_weighted_blocking(curr_dict))
                self.trans_arr = np.append(self.trans_arr, curr_dict['stats']['misc_info']['av_transponders'])
                self.dist_arr = np.append(self.dist_arr, curr_dict['stats']['misc_info']['dist_block'])
                self.cong_arr = np.append(self.cong_arr, curr_dict['stats']['misc_info']['cong_block'])
                # TODO: Change (dist block)
                # self.cong_arr = np.append(self.cong_arr, 100)

                bandwidth_obj = curr_dict['stats']['misc_info']['blocking_obj']
                total_blocks = bandwidth_obj['50'] + bandwidth_obj['100'] + bandwidth_obj['400']

                if total_blocks == 0:
                    self.band_one = np.append(self.band_one, 0)
                    self.band_two = np.append(self.band_two, 0)
                    self.band_three = np.append(self.band_three, 0)
                else:
                    self.band_one = np.append(self.band_one, ((float(bandwidth_obj['50']) / total_blocks) * 100.0))
                    self.band_two = np.append(self.band_two, ((float(bandwidth_obj['100']) / total_blocks) * 100.0))
                    self.band_three = np.append(self.band_three, ((float(bandwidth_obj['400']) / total_blocks) * 100.0))

                if erlang == 50:
                    self.mu = curr_dict['stats']['misc_info']['mu']
                    self.num_cores = curr_dict['stats']['misc_info']['cores_used']
                    self.spectral_slots = curr_dict['stats']['misc_info']['spectral_slots']
                    if self.lps:
                        max_lps = curr_dict['stats']['misc_info']['max_lps']
                    else:
                        max_lps = None

                # Only plot up to Erlang 400, beyond that is considered not important for the time being
                if erlang == 400:
                    break

            self.plot_dict[thread_num] = dict()
            self.plot_dict[thread_num]['erlang'] = self.erlang_arr
            self.plot_dict[thread_num]['blocking'] = self.blocking_arr
            self.plot_dict[thread_num]['av_trans'] = self.trans_arr
            self.plot_dict[thread_num]['max_lps'] = max_lps
            self.plot_dict[thread_num]['cong_block'] = self.cong_arr
            self.plot_dict[thread_num]['dist_block'] = self.dist_arr
            self.plot_dict[thread_num]['50'] = self.band_one
            self.plot_dict[thread_num]['100'] = self.band_two
            self.plot_dict[thread_num]['400'] = self.band_three

            self.erlang_arr = np.array([])
            self.blocking_arr = np.array([])
            self.trans_arr = np.array([])
            self.dist_arr = np.array([])
            self.cong_arr = np.array([])
            self.band_one = np.array([])
            self.band_two = np.array([])
            self.band_three = np.array([])
            self.erlang_arr = np.array([])


if __name__ == '__main__':
    blocking_obj = Blocking()
    blocking_obj.get_data()
    blocking_obj.plot_blocking()
    blocking_obj.plot_transponders()
    blocking_obj.plot_block_percents()
    blocking_obj.plot_bandwidths()
