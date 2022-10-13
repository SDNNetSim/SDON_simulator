import os
import json
import numpy as np
import matplotlib.pyplot as plt


class Blocking:
    """
    Creates and saves plot of blocking percentage vs. Erlang.
    """

    def __init__(self):
        self.file_path = '../data/output/'
        self.files = self.get_file_names()
        self.erlang_arr = np.array([])
        self.blocking_arr = np.array([])

    def get_file_names(self):
        """
        Obtains all the filenames of the output data for each Erlang.
        """
        return sorted([f for f in os.listdir(self.file_path) if os.path.isfile(os.path.join(self.file_path, f))])

    def plot_blocking_means(self):
        """
        Plots blocking means vs. Erlang values.
        """
        for file_name in self.files:
            with open(self.file_path + file_name, 'r', encoding='utf-8') as curr_f:
                curr_dict = json.load(curr_f)

            erlang = float(file_name.split('_')[0])
            blocking_mean = curr_dict['stats']['mean']
            if blocking_mean is None:
                blocking_mean = 0
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
        plt.plot(self.erlang_arr, self.blocking_arr)
        plt.show()


if __name__ == '__main__':
    blocking_obj = Blocking()
    blocking_obj.plot_blocking_means()
