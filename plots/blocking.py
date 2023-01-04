import os
import json
import numpy as np
import matplotlib.pyplot as plt

# TODO: Label Axis and label What mu was equal to


class Blocking:
    """
    Creates and saves plot of blocking percentage vs. Erlang.
    """

    def __init__(self):
        self.file_path = '../data/output'
        self.files = self.get_file_names()
        self.erlang_arr = np.array([])
        self.blocking_arr = np.array([])

    def get_file_names(self):
        """
        Obtains all the filenames of the output data for each Erlang.
        """
        return sorted([float(f.split('_')[0]) for f in os.listdir(self.file_path)
                       if os.path.isfile(os.path.join(self.file_path, f))])

    def plot_blocking_means(self):
        """
        Plots blocking means vs. Erlang values.
        """
        for erlang in self.files:
            with open(f'{self.file_path}/{erlang}_erlang.json', 'r', encoding='utf-8') as curr_f:
                curr_dict = json.load(curr_f)

            # TODO: Change this before plotting!
            blocking_mean = curr_dict['stats']['mean']
            if blocking_mean is None:
                blocking_mean = 0
                # continue
            else:
                blocking_mean = float(blocking_mean)

            # TODO: Make sure this appends in order?
            self.erlang_arr = np.append(self.erlang_arr, erlang)
            self.blocking_arr = np.append(self.blocking_arr, blocking_mean)

        self.save_plot()

    def save_plot(self):
        """
        Saves and shows the plot.
        """
        # TODO: Update to save
        # TODO: Add grids
        plt.yscale('log')
        # self.erlang_arr = [erlang for erlang in range(50, 800, 50)]
        # self.blocking_arr = [0, 0, 0, 0, 0, 0, 0, 0.0013, 0.0084, 0.0158, 0.0242, 0.0314, 0.0423, 0.0467, 0.0575]
        # if len(self.erlang_arr) != len(self.blocking_arr):
        #     raise ValueError

        plt.plot(self.erlang_arr, self.blocking_arr)
        plt.grid()
        plt.yticks([10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1])
        plt.show()


if __name__ == '__main__':
    blocking_obj = Blocking()
    blocking_obj.plot_blocking_means()
    blocking_obj.save_plot()
