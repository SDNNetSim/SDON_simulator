import os
import json
import numpy as np
import matplotlib.pyplot as plt  # pylint: disable=import-error

from useful_functions.handle_dirs_files import create_dir


class Blocking:
    """
    Creates and saves plot of blocking percentage vs. Erlang.
    """

    def __init__(self):
        # Change these variables for the desired plot you'd like
        # TODO: Document the structure of how things are saved
        # TODO: Default to latest one if none is chosen (mark this on the graph)
        self.des_times = ['0201_10:51:47', '0201_10:55:05']
        self.network_name = 'USNet'
        self.base_path = f'../data/output/{self.network_name}'
        self.files = self.get_file_names()

        self.erlang_arr = np.array([])
        self.blocking_arr = np.array([])
        self.plot_dict = dict()
        self.mu = None  # pylint: disable=invalid-name
        self.num_cores = None
        self.spectral_slots = None
        self.max_lps = None
        self.trans_arr = np.array([])

        self.lps = True

    def get_file_names(self):
        """
        Obtains all the filenames of the output data for each Erlang.
        """
        res_dict = dict()
        for curr_time in self.des_times:
            curr_fp = f'{self.base_path}/{curr_time}/'
            tmp_list = sorted([float(f.split('_')[0]) for f in os.listdir(curr_fp)
                               if os.path.isfile(os.path.join(curr_fp, f))])
            res_dict[curr_time] = tmp_list

        return res_dict

    def plot_blocking_means(self):
        """
        Plots blocking means vs. Erlang values.
        """
        for curr_time, lst in self.files.items():
            self.plot_dict[curr_time] = dict()

            for erlang in lst:
                curr_fp = f'{self.base_path}/{curr_time}/'
                with open(f'{curr_fp}/{erlang}_erlang.json', 'r', encoding='utf-8') as curr_f:
                    curr_dict = json.load(curr_f)

                blocking_mean = curr_dict['stats']['mean']
                # Only one iteration occurred, no mean calculated for now
                if blocking_mean is None:
                    blocking_mean = curr_dict['simulations']['0']

                blocking_mean = float(blocking_mean)

                self.erlang_arr = np.append(self.erlang_arr, erlang)
                self.blocking_arr = np.append(self.blocking_arr, blocking_mean)
                self.trans_arr = np.append(self.trans_arr,
                                           curr_dict['stats']['misc_info']['av_transponders'])

                if erlang == 50:
                    self.mu = curr_dict['stats']['misc_info']['mu']
                    self.num_cores = curr_dict['stats']['misc_info']['cores_used']
                    self.spectral_slots = curr_dict['stats']['misc_info']['spectral_slots']
                    if self.lps:
                        self.max_lps = curr_dict['stats']['misc_info']['max_lps']

            self.plot_dict[curr_time]['erlang'] = self.erlang_arr
            self.plot_dict[curr_time]['blocking'] = self.blocking_arr
            self.plot_dict[curr_time]['max_lps'] = self.max_lps
            self.plot_dict[curr_time]['av_trans'] = self.trans_arr

            self.erlang_arr = np.array([])
            self.blocking_arr = np.array([])
            self.trans_arr = np.array([])

        self.save_plot()

    def plot_transponders(self):
        for curr_time, lst in self.files.items():
            self.plot_dict[curr_time] = dict()

            for erlang in lst:
                curr_fp = f'{self.base_path}/{curr_time}/'
                with open(f'{curr_fp}/{erlang}_erlang.json', 'r', encoding='utf-8') as curr_f:
                    curr_dict = json.load(curr_f)

                self.erlang_arr = np.append(self.erlang_arr, erlang)
                self.trans_arr = np.append(self.trans_arr,
                                           curr_dict['stats']['misc_info']['av_transponders'])
                self.max_lps = curr_dict['stats']['misc_info']['max_lps']

            self.plot_dict[curr_time]['erlang'] = self.erlang_arr
            self.plot_dict[curr_time]['max_lps'] = self.max_lps
            self.plot_dict[curr_time]['av_trans'] = self.trans_arr

            self.erlang_arr = np.array([])
            self.trans_arr = np.array([])

        self.save_plot(plot_trans=True)

    def save_plot(self, plot_trans=False):
        """
        Saves and shows the plot.
        """
        # TODO: Blocking and transponders should be two separate graphs
        if plot_trans:
            plt.title(f'{self.network_name} Transponders vs. Erlang (Core = {self.num_cores})')
            plt.ylabel('Transponders per Request')
        else:
            plt.title(f'{self.network_name} BP vs. Erlang (Core = {self.num_cores})')
            plt.yscale('log')
            plt.ylabel('Blocking Probability')

        plt.grid()
        plt.xlabel('Erlang')

        create_dir(f'./output/{self.network_name}')

        legend_list = list()
        for curr_time, obj in self.plot_dict.items():
            if plot_trans:
                plt.plot(obj['erlang'], obj['av_trans'])
            else:
                plt.plot(obj['erlang'], obj['blocking'])

            legend_list.append(f"LS = {obj['max_lps']}")

        if not plot_trans:
            plt.yticks([10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1])

        plt.xticks([erlang for erlang in range(10, 810, 100)])
        plt.legend(legend_list)

        # Always saves based on the last time in the list
        plt.savefig(f'./output/{self.network_name}/{self.des_times[-1]}.png')
        plt.show()


if __name__ == '__main__':
    blocking_obj = Blocking()
    # blocking_obj.plot_blocking_means()
    blocking_obj.plot_transponders()
