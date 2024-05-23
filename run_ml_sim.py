import os

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import joblib

from config_scripts.parse_args import parse_args
from config_scripts.setup_config import read_config
from helper_scripts.ml_helpers import process_data


# TODO: This is the main script that will run the machine learning simulation.
#   - Helper functions in ml_helpers.py to set up models
#   - Any arguments for this script in ml_args.py
#   - The scikit learn library does it's thing
#   - I control the result output and saving.
# TODO: Is training is probably not needed.
def _run_iters():
    """
    Handles the main training or testing iterations.

    :return:
    """
    raise NotImplementedError


def _get_model():
    pass


def _print_info():
    pass


# TODO: I'm not sure if we need an environment object here.
# TODO: The if-else logic for train/test will be implemented in the _run function.
#   - Getting a trained model
#   - Running the trained model
#   - Else, training the model
#   - Calls _run_iters
def _run(sim_dict: dict):
    """
    Controls the simulation of the machine learning model.

    :return: None
    """
    # TODO: Only support for running one process.
    sim_dict = sim_dict['s1']

    base_fp = 'data/output/'
    train_fp = os.path.join(base_fp, sim_dict['train_file_path'])
    train_fp = os.path.join(train_fp, '700.0_train_data.csv')
    if sim_dict['is_training']:
        df = pd.read_csv(train_fp)

        df_processed = process_data(input_df=df)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_processed)

        X_train, X_val = train_test_split(X_scaled, test_size=0.3, random_state=42)

        kmeans = KMeans(n_clusters=3, random_state=0)
        kmeans.fit(X_train)

        inertia = kmeans.inertia_
        print(f"Inertia: {inertia}")

        silhouette_avg = silhouette_score(X_val, kmeans.predict(X_val))
        print(f"Silhouette Score: {silhouette_avg}")
    else:
        raise NotImplementedError


def _setup_ml_sim():
    """
    Gets the simulation input parameters.

    :return: The simulation input parameters.
    :rtype: dict
    """
    args_obj = parse_args()
    config_path = os.path.join('ini', 'run_ini', 'config.ini')
    sim_dict = read_config(args_obj=args_obj, config_path=config_path)

    return sim_dict


def run_ml_sim():
    """
    Controls the simulation of the machine learning model.

    :return: None
    """
    sim_dict = _setup_ml_sim()
    _run(sim_dict=sim_dict)


if __name__ == '__main__':
    run_ml_sim()
