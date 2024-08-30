from run_sim import run
from config_scripts.parse_args import parse_args
from config_scripts.setup_config import read_config
import json
import sys
import os

args_obj = parse_args()

def run_comparison():
    all_sims_dict = read_config(args_obj=args_obj, config_path=args_obj['config_path'])

    # Call the run function from run_sim.py
    run(sims_dict=all_sims_dict)

def find_sim_type():
    file_type = read_config(args_obj=args_obj, config_path=args_obj['config_path'])
    if general_settings in file_type:
        if sim_type in file_type[general_settings]:
            return file_type[general_settings][sim_type]
        else:
            return ValueError("Missing simulation type in configuration file.")
    else:
        return ValueError("Missing simulation type parameter in configuration file under general_settings")

def compare_json_files(old_file, new_file):
    """Load and compare two JSON files."""
    with open(old_file, 'r') as f:
        old_data = json.load(f)

    with open(new_file, 'r') as g:
        new_data = json.load(g)

    if old_data == new_data:
        print("The comparison results pass.")
    else:
        print("The comparison results do not pass.")
        sys.exit(1)


def find_newest_file(directory):
    newest_file = None
    latest_time = 0

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        mod_time = os.path.getmtime(file_path)
        if mod_time > latest_time:
            latest_time = mod_time
            newest_file = filename

    newest_file = directory + '/' + newest_file
    return newest_file


if __name__ == "__main__":
    run_comparison()
    ##simulation_test_type = find_sim_type()        TODO: fix find_sim_type so it can read which is which
    old_saved_data_path = './data/run_comparison_data/yue_run_data.json'
    path_to_output = './data/output/NSFNet'
    date_of_simulation = find_newest_file(path_to_output)
    time_of_data_path = find_newest_file(date_of_simulation)
    add_simulation_run_to_path = time_of_data_path + '/s1'
    new_saved_data_path = find_newest_file(add_simulation_run_to_path)
    print(new_saved_data_path)
    compare_json_files(old_saved_data_path, new_saved_data_path)