from scripts.structure_raw_data import structure_data


class RunSim:
    """
    Runs the simulations for this project.
    """

    def __init__(self):
        self.seed = list()
        self.hold_time_mean = 3600
        self.inter_arrival_time = 10
        self.number_of_request = 1000

        self.bw_type = {
            "100 Gbps": {
                "DP-QPSK": 3
            },
            "400 Gbps": {
                "DP-QPSK": 10
            }
        }

        self.num_iteration = 10
        self.core_slots_num = 256

    def save_input(self):
        pass

    def create_pt(self):
        pass

    def create_input(self):
        """
        Creates input data.
        """
        pass

    def run(self):
        """
        Controls the class.
        """
        pass


# TODO: Find a better way to organize data directories and name them
# TODO: Update documentation
if __name__ == '__main__':
    pass
