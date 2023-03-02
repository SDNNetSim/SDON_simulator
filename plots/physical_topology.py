import networkx as nx
import matplotlib.pyplot as plt  # pylint: disable=import-error


class PhysicalTopology:  # pylint: disable=too-few-public-methods
    """
    Creates and saves plot of physical topology.
    """

    def __init__(self, physical_topology, show_plot=True):
        self.physical_topology = physical_topology
        self.show_plot = show_plot

    def plot_pt(self):
        """
        Plots the physical topology. A graph of nodes and links.
        """
        nx.draw(self.physical_topology, with_labels=True)
        plt.savefig("../assets/physical_topology.png")

        if self.show_plot:
            plt.show()


if __name__ == '__main__':
    pt_obj = PhysicalTopology(physical_topology=None)
    pt_obj.plot_pt()
