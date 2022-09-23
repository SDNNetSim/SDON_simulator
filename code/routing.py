import networkx as nx


def routing(source, destination, physical_topology, slots_dictionary):  # pylint: disable=unused-argument
    path = nx.shortest_path(G=physical_topology, source=source, target=destination)
    return path
