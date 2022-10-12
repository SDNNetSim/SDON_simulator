import os
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt



class Physical_Topology:
    """
    Creates and saves plot of physical topology.
    """
    def __init__(self):
        self.file_path = '../data/input/'
        self.files = self.get_file_names()

    def get_file_names(self):
        return sorted([f for f in os.listdir(self.file_path) if os.path.isfile(os.path.join(self.file_path, f))])

    def plot_PT(self):
        nx.draw(self.physical_topology, with_labels = True)
        plt.savefig(self.files + ".png")
        plt.show()



if __name__ == '__main__':
    Physical_Topology_obj = Physical_Topology()
    Physical_Topology_obj.plot_PT()
