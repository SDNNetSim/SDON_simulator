import json
import os


def calculate_mean_from_json(file_path, key):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        iters_list = list(data['iter_stats'].keys())
        last_iter = iters_list[-1]
        if key in data['iter_stats'][last_iter]:
            values = data['iter_stats'][last_iter][key]
            if len(values) != 10:
                raise ValueError('Number of iterations is less than 10.')

            if isinstance(values, list) and all(isinstance(item, (int, float)) for item in values):
                mean_value = sum(values) / len(values)
                return mean_value
            else:
                raise ValueError(f"The key '{key}' does not contain a list of numbers.")
        else:
            raise KeyError(f"The key '{key}' was not found in the JSON file.")

    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading JSON file: {e}")
    except KeyError as ke:
        print(ke)
    except ValueError as ve:
        print(ve)
    return None


def calculate_means_in_directory(directory_path, key):
    mean_values = []
    try:
        for filename in os.listdir(directory_path):
            if filename.endswith(".json"):
                file_path = os.path.join(directory_path, filename)
                mean_value = calculate_mean_from_json(file_path, key)

                if mean_value is not None:
                    mean_values.append((filename, mean_value))

    except Exception as e:
        print(f"An error occurred: {e}")

    return mean_values


directory_path = '../data/output/USNet/1018/12_41_42_158677/s2'
key = 'sim_block_list'
means_list = calculate_means_in_directory(directory_path, key)

for filename, mean_value in means_list:
    print(f"File: {filename} | Mean of '{key}': {mean_value}")
