import networkx as nx


def routing(source, destination, physical_topology, slots_dictionary):
    path = nx.shortest_path(G=physical_topology, source=source, target=destination)
    return path
