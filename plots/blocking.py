import os
import json
import numpy as np
import matplotlib.pyplot as plt  # pylint: disable=import-error


class Blocking:
    """
    Creates and saves plot of blocking percentage vs. Erlang.
    """

    def __init__(self):
        # Change these variables for the desired plot you'd like
        # TODO: Document the structure of how things are saved
        # TODO: Default to latest one if none is chosen (mark this on the graph)
        # TODO: Document how much LPS was used
        self.des_time = '0110_10:13:17'
        self.network_name = 'USNet'
        self.file_path = f'../data/output/{self.network_name}/{self.des_time}/'
        self.files = self.get_file_names()

        self.erlang_arr = np.array([])
        self.blocking_arr = np.array([])
        self.mu = None  # pylint: disable=invalid-name
        self.num_cores = None
        self.spectral_slots = None

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

            blocking_mean = curr_dict['stats']['mean']
            # Only one iteration occurred, no mean calculated for now
            if blocking_mean is None:
                blocking_mean = curr_dict['simulations']['0']

            blocking_mean = float(blocking_mean)

            self.erlang_arr = np.append(self.erlang_arr, erlang)
            self.blocking_arr = np.append(self.blocking_arr, blocking_mean)

            if erlang == 50:
                self.mu = curr_dict['stats']['misc_info']['mu']
                self.num_cores = curr_dict['stats']['misc_info']['cores_used']
                self.spectral_slots = curr_dict['stats']['misc_info']['spectral_slots']

        self.save_plot()

    def save_plot(self):
        """
        Saves and shows the plot.
        """
        plt.yscale('log')
        plt.plot(self.erlang_arr, self.blocking_arr)

        plt.grid()
        plt.legend([f"Cores = {self.num_cores} Mu = {self.mu} Spectral Slots = {self.spectral_slots}"])

        plt.title(f'{self.network_name} BP vs. Erlang')
        plt.xlabel('Erlang')
        plt.ylabel('Blocking Probability')
        plt.yticks([10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1])

        if not os.path.exists('./output/'):
            os.mkdir('./output/')
        if not os.path.exists(f'./output/{self.network_name}'):
            os.mkdir(f'./output/{self.network_name}')

        plt.savefig(f'./output/{self.network_name}/{self.des_time}.png')

        plt.show()


if __name__ == '__main__':
    blocking_obj = Blocking()
    blocking_obj.plot_blocking_means()
    blocking_obj.save_plot()
