import json


def load_input():
    with open('../data/input.json') as json_file:
        sim_input = json.load(json_file)
    return sim_input
