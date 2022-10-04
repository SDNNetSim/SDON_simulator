import networkx as nx
import numpy as np


# TODO: Ask about slots dictionary
class Routing:
    def __init__(self, source, destination, physical_topology, slots_needed=None):
        self.path = None

        self.source = source
        self.destination = destination
        self.physical_topology = physical_topology
        self.slots_needed = slots_needed

    def nx_shortest_path(self):
        return nx.shortest_path(G=self.physical_topology, source=self.source, target=self.destination)

    def least_congested_path(self):
        """
        - Implement
        - Figure out what the first method returns (its type) and return the same
        - Fix engine to run on this (debug on both methods)
        :return:
        """
        count = 0

        while True:
            count += 1
            if count == 1:
                pass
            else:
                pass
