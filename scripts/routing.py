import networkx as nx


# TODO: Ask about slots_dictionary
def routing(source, destination, physical_topology, slots_dictionary):  # pylint: disable=unused-argument
    """
    Computes the shortest path in the graph given the source and destination.

    :param source: The source location
    :type source: str
    :param destination: The destination location
    :type destination: str
    :param physical_topology: Variables related to the physical topology
    :type physical_topology: dict
    :param slots_dictionary: Not sure what this is yet
    :return: The shortest path found
    :rtype: list
    """
    path = nx.shortest_path(G=physical_topology, source=source, target=destination)
    return path
