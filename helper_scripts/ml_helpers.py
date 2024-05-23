import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def load_model():
    """
    Loads a previously trained machine learning model.
    :return:
    """
    raise NotImplementedError


def save_model():
    """
    Saves a trained machine learning model.
    :return:
    """
    raise NotImplementedError


def get_kmeans_stats():
    """
    Get statistics for KMeans clustering.
    """
    # inertia = kmeans.inertia_
    # print(f"Inertia: {inertia}")

    # silhouette_avg = silhouette_score(X_val, kmeans.predict(X_val))
    # print(f"Silhouette Score: {silhouette_avg}")
    raise NotImplementedError


def process_data(input_df: pd.DataFrame):
    """
    Process data for machine learning model.

    :param input_df: Input dataframe.
    :return: Modified processed dataframe.
    :rtype: pd.DataFrame
    """
    input_df['mod_format'] = input_df['mod_format'].str.replace('-', '')
    df_processed = pd.get_dummies(input_df, columns=['bandwidth', 'mod_format'])

    for col in df_processed.columns:
        if df_processed[col].dtype == bool:
            df_processed[col] = df_processed[col].astype(int)

    scaler = StandardScaler()
    feat_scale_list = ['path_length']
    df_processed[feat_scale_list] = scaler.fit_transform(df_processed[feat_scale_list])

    # TODO: Can't input matrix for kmeans but for other algorithms?
    # num_cores = len(df['spec_util_matrix'][0])
    # for core_index in range(num_cores):
    #     column_name = f'core_{core_index}'
    #     df_processed[column_name] = df_processed['spec_util_matrix'].apply(lambda x: x[core_index])
    #
    # matrix_columns = [col for col in df_processed.columns if col.startswith('core_')]
    # for col in matrix_columns:
    #     df_processed[col] = df_processed[col].apply(lambda x: [float(i) for i in x])

    df_processed = df_processed.drop(columns=['spec_util_matrix', 'num_slices'])

    return df_processed


# TODO: Save results
def plot_clusters(df_pca: pd.DataFrame, kmeans: object):
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
