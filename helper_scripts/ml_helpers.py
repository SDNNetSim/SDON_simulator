import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import silhouette_score

from helper_scripts.os_helpers import create_dir
from helper_scripts.sim_helpers import find_path_len, find_core_cong


# TODO: Double check this function as well
def _get_ml_obs(tmp_dict: dict, engine_props: dict, sdn_props: dict):
    df_processed = pd.DataFrame(tmp_dict, index=[0])
    df_processed = pd.get_dummies(df_processed, columns=['old_bandwidth'])

    for col in df_processed.columns:
        if df_processed[col].dtype == bool:
            df_processed[col] = df_processed[col].astype(int)

    for bandwidth, percent in engine_props['request_distribution'].items():
        if percent > 0:
            if bandwidth != sdn_props['bandwidth']:
                df_processed[f'old_bandwidth_{bandwidth}'] = 0

    column_order_list = ['path_length', 'longest_reach', 'ave_cong', 'old_bandwidth_50',
                         'old_bandwidth_100', 'old_bandwidth_200', 'old_bandwidth_400']
    df_processed = df_processed.reindex(columns=column_order_list)

    return df_processed


def get_ml_obs(req_dict: dict, engine_props: dict, sdn_props: dict):
    path_length = find_path_len(path_list=sdn_props['path_list'], topology=engine_props['topology'])
    cong_arr = np.array([])
    # TODO: Repeat code
    for core_num in range(engine_props['cores_per_link']):
        curr_cong = find_core_cong(core_index=core_num, net_spec_dict=sdn_props['net_spec_dict'],
                                   path_list=sdn_props['path_list'])
        cong_arr = np.append(cong_arr, curr_cong)

    tmp_dict = {
        'old_bandwidth': req_dict['bandwidth'],
        'path_length': path_length,
        'longest_reach': req_dict['mod_formats']['QPSK']['max_length'],
        'ave_cong': float(np.mean(cong_arr)),
    }

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


# TODO: No scaling no 50/50 split
# TODO: No scaling w/ 50/50 split
# TODO: Scaling no 50/50 split
# TODO: Scaling w/ 50/50 split
def process_data(input_df: pd.DataFrame):
    """
    Process data for machine learning model.

    :param input_df: Input dataframe.
    :return: Modified processed dataframe.
    :rtype: pd.DataFrame
    """
    df_processed = pd.get_dummies(input_df, columns=['old_bandwidth'])

    for col in df_processed.columns:
        if df_processed[col].dtype == bool:
            df_processed[col] = df_processed[col].astype(int)

    return df_processed


def even_process_data(input_df: pd.DataFrame):
    """
    Process data for machine learning model.

    :param input_df: Input dataframe.
    :return: Modified processed dataframe.
    :rtype: pd.DataFrame
    """
    df1 = input_df[input_df['num_segments'] == 1]
    df2 = input_df[input_df['num_segments'] > 1]
    min_size = min(len(df1), len(df2))

    df1 = df1.sample(n=min_size, random_state=42)
    df2 = df2.sample(n=min_size, random_state=42)
    df_processed = pd.concat([df1, df2])
    df_processed = df_processed.sample(frac=1, random_state=42)
    df_processed = pd.get_dummies(df_processed, columns=['old_bandwidth'])

    for col in df_processed.columns:
        if df_processed[col].dtype == bool:
            df_processed[col] = df_processed[col].astype(int)

    return df_processed


def plot_confusion(sim_dict: dict, y_test, y_pred, erlang: str, algorithm: str):
    """
    Plots a confusion matrix and prints out the accuracy, precision, recall, and F1 score.

    :param sim_dict: The simulation dictionary.
    :param y_test: Testing data.
    :param y_pred: Predictions.
    :param erlang: The Erlang value.
    :param algorithm: The algorithm used.
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f_score = f1_score(y_test, y_pred, average='weighted')

    labels = np.unique(np.concatenate((y_test, y_pred)))
    # Plot a confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(10, 8), dpi=300)  # Increase the quality by increasing dpi
    sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {algorithm}', weight='bold')
    plt.xlabel('Predicted Segments', weight='bold')
    plt.ylabel('Actual Segments', weight='bold')

    # Add accuracy, precision, recall, and F1 score to the plot
    plt.text(0.5, 1.1, f'Accuracy: {accuracy:.5f}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.5, 1.2, f'Precision: {precision:.5f}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.5, 1.3, f'Recall: {recall:.5f}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.5, 1.4, f'F1 Score: {f_score:.5f}', fontsize=12, transform=plt.gca().transAxes)

    save_fp = os.path.join('data', 'plots', sim_dict['train_file_path'])
    create_dir(file_path=save_fp)

    save_fp = os.path.join(save_fp, f'confusion_matrix_{erlang}.png')
    plt.savefig(save_fp, bbox_inches='tight')

    plt.show()


def plot_2d_clusters(df_pca: pd.DataFrame):
    """
    Plot the test data points and their predicted labels.

    :param df_pca: A dataframe normalized with PCA.
    """
    plt.figure(figsize=(10, 8))

    # Plot the test data points, colored by their predicted labels
    scatter = plt.scatter(df_pca["PC1"], df_pca["PC2"], c=df_pca["predicted_label"], cmap='Set1')
    plt.title("KNN Predicted Labels (PCA-reduced Data)")
    plt.xlabel("Principal Component 1 (PC1)")
    plt.ylabel("Principal Component 2 (PC2)")
    plt.colorbar(scatter, label='num_slices')
    plt.show()


def plot_3d_clusters(df_pca: pd.DataFrame):
    """
    Plot the test data points and their predicted labels in 3D.

    :param df_pca: A dataframe normalized with PCA.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the test data points, colored by their predicted labels
    scatter = ax.scatter(df_pca["PC1"], df_pca["PC2"], df_pca["PC3"], c=df_pca["predicted_label"], cmap='Set1')
    ax.set_title("KNN Predicted Labels (PCA-reduced Data)")
    ax.set_xlabel("Principal Component 1 (PC1)")
    ax.set_ylabel("Principal Component 2 (PC2)")
    ax.set_zlabel("Principal Component 3 (PC3)")
    fig.colorbar(scatter, label='num_slices')
    plt.show()
