import json
def load_input():
    input = {}
    with open('input.json') as json_file:
        input = json.load(json_file)
    return input
    