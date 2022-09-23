import json


def load_input():
    with open('../data/input.json', encoding='utf-8') as json_file:
        sim_input = json.load(json_file)
    return sim_input
