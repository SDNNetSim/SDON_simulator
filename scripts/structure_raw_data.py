import pandas as pd

PAIRINGS_FILE_PATH = '../data/raw/europe_network.xlsx'
LINK_LEN_FILE_PATH = '../data/raw/network_distance.txt'


def assign_link_lengths(node_pairings=None):
    return NotImplementedError


def create_node_pairs():
    """
    Takes an Excel file where node numbers are mapped to actual names and creates a dictionary of
    node number to node name pairs.

    :return: A dictionary of each node number to a node name
    :rtype: dict
    """
    df = pd.read_excel(PAIRINGS_FILE_PATH)
    print(df.tail())

    return NotImplementedError


def structure_data():
    tmp_resp = create_node_pairs()
    return assign_link_lengths(node_pairings=tmp_resp)


if __name__ == '__main__':
    structure_data()
