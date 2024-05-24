import os
import ast

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# TODO: Save model
import joblib

from config_scripts.parse_args import parse_args
from config_scripts.setup_config import read_config
from helper_scripts.ml_helpers import process_data, plot_clusters, plot_confusion


def _print_info():
    pass


def _handle_testing():
    raise NotImplementedError


def _handle_training(sim_dict: dict, file_path: str):
    data_frame = pd.read_csv(file_path, converters={'spec_util_matrix': ast.literal_eval})
    df_processed = process_data(input_df=data_frame)

    # TODO: Generalize test set size and number of pca components
    x_train, x_vals = train_test_split(df_processed, test_size=0.3, random_state=42)

    # TODO: Split to other functions eventually?
    if sim_dict['ml_model'] == 'kmeans':
        pca = PCA(n_components=2)
        x_pca = pca.fit_transform(x_train)
        kmeans = KMeans(n_clusters=8, random_state=0)
        kmeans.fit(x_pca)

        df_pca = pd.DataFrame(data=x_pca, columns=["PC1", "PC2"])
        df_pca["cluster"] = kmeans.labels_
        df_pca["true_label"] = df_processed['num_slices']
        plot_clusters(df_pca=df_pca, kmeans=kmeans)
    elif sim_dict['ml_model'] == 'logistic_regression':
        predictor_df = df_processed['num_slices']
        feature_df = df_processed.drop('num_slices', axis=1)

        x_train, x_test, y_train, y_test = train_test_split(feature_df, predictor_df, test_size=0.3, random_state=42)
        lr_obj = LogisticRegression(random_state=0)
        lr_obj.fit(x_train, y_train)
        y_pred = lr_obj.predict(x_test)
        plot_confusion(y_test=y_test, y_pred=y_pred)
    else:
        raise NotImplementedError


def _run(sim_dict: dict):
    """
    Controls the simulation of the machine learning model.

    :return: None
    """
    # TODO: Only support for running one process.
    sim_dict = sim_dict['s1']

    base_fp = 'data/output/'
    if sim_dict['is_training']:
        train_fp = os.path.join(base_fp, sim_dict['train_file_path'])
        # TODO: We need Erlang here (generalize)
        train_fp = os.path.join(train_fp, '700.0_train_data.csv')
        _handle_training(sim_dict=sim_dict, file_path=train_fp)
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
