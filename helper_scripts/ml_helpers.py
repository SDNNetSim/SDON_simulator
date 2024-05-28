import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import silhouette_score

from helper_scripts.os_helpers import create_dir
from helper_scripts.sim_helpers import find_path_len, find_core_cong, get_path_mod


# TODO: Double check this function as well
def _get_ml_obs(tmp_dict: dict, engine_props: dict, sdn_props: dict):
    df_processed = pd.DataFrame(tmp_dict, index=[0])
    df_processed = pd.get_dummies(df_processed, columns=['bandwidth'])

    for col in df_processed.columns:
        if df_processed[col].dtype == bool:
            df_processed[col] = df_processed[col].astype(int)

    for bandwidth, percent in engine_props['request_distribution'].items():
        if percent > 0:
            if bandwidth != sdn_props['bandwidth']:
                df_processed[f'bandwidth_{bandwidth}'] = 0

    column_order_list = ['path_length', 'ave_cong', 'longest_reach', 'bandwidth_100', 'bandwidth_200',
                         'bandwidth_400']
    df_processed = df_processed.reindex(columns=column_order_list)

    return df_processed


def get_ml_obs(engine_props: dict, sdn_props: dict):
    path_length = find_path_len(path_list=sdn_props['path_list'], topology=engine_props['topology'])
    cong_arr = np.array([])
    # TODO: Repeat code
    for core_num in range(engine_props['cores_per_link']):
        curr_cong = find_core_cong(core_index=core_num, net_spec_dict=sdn_props['net_spec_dict'],
                                   path_list=sdn_props['path_list'])
        cong_arr = np.append(cong_arr, curr_cong)

    # TODO: Make sure you're getting the correct variables here, the above will have to be updated
    tmp_dict = {
        'old_bandwidth': sdn_props['bandwidth'],
        'path_length': path_length,
        'longest_reach': get_path_mod(mods_dict=sdn_props['mod_formats'], path_len=path_length),
        'ave_cong': float(np.mean(cong_arr)),
    }
    if tmp_dict['mod_format'] is False:
        return False

    return _get_ml_obs(engine_props=engine_props, sdn_props=sdn_props, tmp_dict=tmp_dict)


def load_model(engine_props: dict):
    """
    Loads a trained machine learning model.

    :param engine_props: Properties from engine.
    :return: The trained model.
    """

    model_fp = os.path.join('logs', engine_props['ml_model'], engine_props['train_file_path'],
                            f"{engine_props['ml_model']}_{str(int(engine_props['erlang']))}.joblib")
    resp = joblib.load(filename=model_fp)

    return resp


def save_model(sim_dict: dict, model, algorithm: str, erlang: str):
    """
    Saves a trained machine learning model.

    :param sim_dict: The simulation dictionary.
    :param model: The trained model.
    :param algorithm: The filename to save the model as.
    :param erlang: The Erlang value.
    """
    base_fp = os.path.join('logs', algorithm, sim_dict['train_file_path'])
    create_dir(file_path=base_fp)

    save_fp = os.path.join(base_fp, f'{algorithm}_{erlang}.joblib')
    joblib.dump(model, save_fp)


def get_kmeans_stats(kmeans: object, x_val):
    """
    Get statistics for KMeans clustering.
    """
    inertia = kmeans.inertia_
    print(f"Inertia: {inertia}")

    silhouette_avg = silhouette_score(x_val, kmeans.predict(x_val))
    print(f"Silhouette Score: {silhouette_avg}")


def process_data(input_df: pd.DataFrame):
    """
    Process data for machine learning model.

    :param input_df: Input dataframe.
    :return: Modified processed dataframe.
    :rtype: pd.DataFrame
    """
    input_df['mod_format'] = input_df['mod_format'].str.replace('-', '')
    df_processed = pd.get_dummies(input_df, columns=['bandwidth'])
    df_processed = df_processed.drop('was_sliced', axis=1)

    for col in df_processed.columns:
        if df_processed[col].dtype == bool:
            df_processed[col] = df_processed[col].astype(int)

    return df_processed


def plot_confusion(sim_dict: dict, y_test, y_pred, erlang: str):
    """
    Plots a confusion matrix and prints out the accuracy, precision, recall, and F1 score.

    :param sim_dict: The simulation dictionary.
    :param y_test: Testing data.
    :param y_pred: Predictions.
    :param erlang: The Erlang value.
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f_score = f1_score(y_test, y_pred, average='weighted')

    # Plot a confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8), dpi=300)  # Increase the quality by increasing dpi
    sns.heatmap(conf_mat, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Add accuracy, precision, recall, and F1 score to the plot
    plt.text(0.5, 1.1, f'Accuracy: {accuracy:.2f}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.5, 1.2, f'Precision: {precision:.2f}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.5, 1.3, f'Recall: {recall:.2f}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.5, 1.4, f'F1 Score: {f_score:.2f}', fontsize=12, transform=plt.gca().transAxes)

    save_fp = os.path.join('data', 'plots', sim_dict['train_file_path'])
    create_dir(file_path=save_fp)

    save_fp = os.path.join(save_fp, f'confusion_matrix_{erlang}.png')
    plt.savefig(save_fp, bbox_inches='tight')

    plt.show()


def plot_2d_clusters(df_pca: pd.DataFrame, kmeans: object):
    """
    Plot the clusters of the KMeans algorithm.

    :param df_pca: A dataframe normalized with PCA.
    :param kmeans: Kmeans algorithm object.
    """
    plt.figure(figsize=(10, 8))

    # Create a scatter plot of the PCA-reduced data, colored by "num_slices" value
    scatter = plt.scatter(df_pca["PC1"], df_pca["PC2"], c=df_pca["true_label"], cmap='Set1')

    # Plot the centroids of the clusters
    centers = kmeans.cluster_centers_
    for i, center in enumerate(centers):
        plt.text(center[0], center[1], f'Center {i}', ha='center', va='center', color='red')
    plt.title("K-Means Clustering Results (PCA-reduced Data)")
    plt.xlabel("Principal Component 1 (PC1)")
    plt.ylabel("Principal Component 2 (PC2)")
    plt.colorbar(scatter, label='num_slices')
    plt.show()


def plot_3d_clusters(df_pca: pd.DataFrame, kmeans: object):
    """
    Plot the clusters of the KMeans algorithm in 3D.

    :param df_pca: A dataframe normalized with PCA.
    :param kmeans: Kmeans algorithm object.
    """
    fig = plt.figure(figsize=(10, 8))
    axis = fig.add_subplot(111, projection='3d')

    # Create a scatter plot of the PCA-reduced data, colored by "true_label" value
    scatter = axis.scatter(df_pca["PC1"], df_pca["PC2"], df_pca["PC3"], c=df_pca["true_label"], cmap='Set1')

    # Plot the centroids of the clusters
    centers = kmeans.cluster_centers_
    for i, center in enumerate(centers):
        axis.text(center[0], center[1], center[2], f'Center {i}', ha='center', va='center', color='red')

    axis.set_title("K-Means Clustering Results (PCA-reduced Data)")
    axis.set_xlabel("Principal Component 1 (PC1)")
    axis.set_ylabel("Principal Component 2 (PC2)")
    axis.set_zlabel("Principal Component 3 (PC3)")
    fig.colorbar(scatter, label='num_slices')
    plt.show()
