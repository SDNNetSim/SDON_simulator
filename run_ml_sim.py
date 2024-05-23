import os
import ast

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import joblib
import seaborn as sns

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


def plot_clusters(df_pca, kmeans):
    plt.figure(figsize=(10, 8))

    # Create a scatter plot of the PCA-reduced data, colored by "num_slices" value
    scatter = plt.scatter(df_pca["PC1"], df_pca["PC2"], c=df_pca["true_label"], cmap='viridis')

    # Plot the centroids of the clusters
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

    plt.title("K-Means Clustering Results (PCA-reduced Data)")
    plt.xlabel("Principal Component 1 (PC1)")
    plt.ylabel("Principal Component 2 (PC2)")
    plt.colorbar(scatter, label='num_slices')
    plt.show()


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
        # df = pd.read_csv(train_fp)
        df = pd.read_csv(train_fp, converters={'spec_util_matrix': ast.literal_eval})
        df_processed = process_data(input_df=df)
        scaler = StandardScaler()
        feat_scale_list = ['path_length']
        df_processed[feat_scale_list] = scaler.fit_transform(df_processed[feat_scale_list])

        # num_cores = len(df['spec_util_matrix'][0])
        # for core_index in range(num_cores):
        #     column_name = f'core_{core_index}'
        #     df_processed[column_name] = df_processed['spec_util_matrix'].apply(lambda x: x[core_index])
        #
        # matrix_columns = [col for col in df_processed.columns if col.startswith('core_')]
        # for col in matrix_columns:
        #     df_processed[col] = df_processed[col].apply(lambda x: [float(i) for i in x])

        df_processed = df_processed.drop(columns=['spec_util_matrix', 'num_slices'])
        X_train, X_val = train_test_split(df_processed, test_size=0.3, random_state=42)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_train)

        kmeans = KMeans(n_clusters=4, random_state=0)
        kmeans.fit(X_pca)

        # inertia = kmeans.inertia_
        # print(f"Inertia: {inertia}")

        # silhouette_avg = silhouette_score(X_val, kmeans.predict(X_val))
        # print(f"Silhouette Score: {silhouette_avg}")

        df_pca = pd.DataFrame(data=X_pca, columns=["PC1", "PC2"])
        df_pca["cluster"] = kmeans.labels_
        df_pca["true_label"] = df['num_slices']
        plot_clusters(df_pca=df_pca, kmeans=kmeans)
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
