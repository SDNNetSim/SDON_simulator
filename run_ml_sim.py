import os
import ast

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from config_scripts.parse_args import parse_args
from config_scripts.setup_config import read_config
from helper_scripts.ml_helpers import process_data, plot_2d_clusters, plot_3d_clusters, plot_confusion
from helper_scripts.ml_helpers import save_model


# TODO: Add metric for feature importance
# TODO: Maybe have YAML's like RL for parameters
def _train_test_kmeans(df_processed: pd.DataFrame, sim_dict: dict):
    x_train, _ = train_test_split(df_processed, test_size=sim_dict['test_size'], random_state=42)
    pca = PCA(n_components=3)
    x_pca = pca.fit_transform(x_train)
    kmeans = KMeans(n_clusters=4, random_state=0)
    kmeans.fit(x_pca)

    df_pca = pd.DataFrame(data=x_pca, columns=["PC1", "PC2", "PC3"])
    df_pca["cluster"] = kmeans.labels_
    df_pca["true_label"] = df_processed['num_segments']
    plot_2d_clusters(df_pca=df_pca, kmeans=kmeans)
    plot_3d_clusters(df_pca=df_pca, kmeans=kmeans)

    save_model(model=kmeans, algorithm='kmeans')


def _train_test_lr(df_processed: pd.DataFrame, sim_dict: dict):
    predictor_df = df_processed['num_segments']
    feature_df = df_processed.drop('num_segments', axis=1)

    x_train, x_test, y_train, y_test = train_test_split(feature_df, predictor_df, test_size=sim_dict['test_size'],
                                                        random_state=42)
    lr_obj = LogisticRegression(random_state=0)
    lr_obj.fit(x_train, y_train)
    y_pred = lr_obj.predict(x_test)
    plot_confusion(y_test=y_test, y_pred=y_pred)

    save_model(model=lr_obj, algorithm='logistic_regression')


def _handle_training(sim_dict: dict, file_path: str):
    data_frame = pd.read_csv(file_path, converters={'spec_util_matrix': ast.literal_eval})
    df_processed = process_data(input_df=data_frame)

    if sim_dict['ml_model'] == 'kmeans':
        _train_test_kmeans(df_processed=df_processed, sim_dict=sim_dict)
    elif sim_dict['ml_model'] == 'logistic_regression':
        _train_test_lr(df_processed=df_processed, sim_dict=sim_dict)
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

    # TODO: Run for all Erlang values
    train_fp = os.path.join(base_fp, sim_dict['train_file_path'])
    train_fp = os.path.join(train_fp, '50.0_train_data.csv')
    _handle_training(sim_dict=sim_dict, file_path=train_fp)


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
