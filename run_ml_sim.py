import os
import ast
import glob

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from config_scripts.parse_args import parse_args
from config_scripts.setup_config import read_config
from helper_scripts.ml_helpers import process_data, plot_2d_clusters, plot_3d_clusters, plot_confusion
from helper_scripts.ml_helpers import save_model


# TODO: Add metric for feature importance
# TODO: Maybe have YAML's like RL for parameters
def _train_test_knn(df_processed: pd.DataFrame, sim_dict: dict, erlang: str):
    x_train, x_test, y_train, y_test = train_test_split(df_processed.drop('old_bandwidth_50', axis=1), df_processed['old_bandwidth_50'], test_size=sim_dict['test_size'], random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)

    y_pred = knn.predict(x_test)
    x_test_with_predictions = x_test.copy()
    x_test_with_predictions['predicted_label'] = y_pred

    pca = PCA(n_components=3)  # Perform PCA with 3 components for 3D plotting
    x_test_pca = pca.fit_transform(x_test_with_predictions.drop('predicted_label', axis=1))
    df_pca = pd.DataFrame(data=x_test_pca, columns=["PC1", "PC2", "PC3"])  # Create a DataFrame with 3 principal components
    df_pca["predicted_label"] = x_test_with_predictions['predicted_label']

    # Check the number of principal components and call the appropriate plotting function
    if pca.n_components_ == 2:
        plot_2d_clusters(df_pca=df_pca)
    elif pca.n_components_ == 3:
        plot_3d_clusters(df_pca=df_pca)
    else:
        print("Cannot plot clusters for more than 3 dimensions.")

    save_model(sim_dict=sim_dict, model=knn, algorithm='knn', erlang=erlang)


def _train_test_lr(df_processed: pd.DataFrame, sim_dict: dict, erlang: str):
    predictor_df = df_processed['num_segments']
    feature_df = df_processed.drop('num_segments', axis=1)

    x_train, x_test, y_train, y_test = train_test_split(feature_df, predictor_df, test_size=sim_dict['test_size'],
                                                        random_state=42)
    lr_obj = LogisticRegression(random_state=0)
    lr_obj.fit(x_train, y_train)
    y_pred = lr_obj.predict(x_test)
    plot_confusion(sim_dict=sim_dict, y_test=y_test, y_pred=y_pred, erlang='All')

    for new_erl in range(50, 650, 50):
        save_model(sim_dict=sim_dict, model=lr_obj, algorithm='logistic_regression', erlang=str(new_erl))


def extract_value(path: str):
    parts = path.split('/')
    filename = parts[-1]
    filename_parts = filename.split('_')
    value = filename_parts[0]
    if '.' in value:
        value = value.split('.')[0]
    return value


def _handle_training(sim_dict: dict, file_path: str, train_dir: str):
    # data_frame = None
    # for erlang in range(50, 650, 50):
    #     curr_fp = os.path.join(train_dir, f'{float(erlang)}_train_data.csv')
    #
    #     if data_frame is None:
    #         data_frame = pd.read_csv(curr_fp)
    #     else:
    #         tmp_df = pd.read_csv(curr_fp)
    #         data_frame = pd.concat([data_frame, tmp_df])
    data_frame = pd.read_csv(file_path)
    df_processed = process_data(input_df=data_frame)

    erlang = extract_value(path=file_path)
    if sim_dict['ml_model'] == 'kmeans':
        _train_test_knn(df_processed=df_processed, sim_dict=sim_dict, erlang=erlang)
    elif sim_dict['ml_model'] == 'logistic_regression':
        _train_test_lr(df_processed=df_processed, sim_dict=sim_dict, erlang=None)
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

    train_dir = os.path.join(base_fp, sim_dict['train_file_path'])
    train_files = glob.glob(os.path.join(train_dir, "*.csv"))

    for train_fp in train_files:
        _handle_training(sim_dict=sim_dict, file_path=train_fp, train_dir=train_dir)


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
