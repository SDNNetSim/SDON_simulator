import json


# TODO: Move this to the engine class


def load_input():
    """
    Load and return the simulation input JSON file.

    :return: The JSON object
    :rtype: dict
    """
    with open('../data/input.json', encoding='utf-8') as json_file:
        sim_input = json.load(json_file)
    return sim_input
