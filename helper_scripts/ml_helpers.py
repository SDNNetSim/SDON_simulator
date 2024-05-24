import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import silhouette_score
import seaborn as sns


# TODO: Save results and models
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
    df_processed = pd.get_dummies(input_df, columns=['bandwidth', 'mod_format'])

    for col in df_processed.columns:
        if df_processed[col].dtype == bool:
            df_processed[col] = df_processed[col].astype(int)

    scaler = StandardScaler()
    feat_scale_list = ['path_length', 'ave_shannon', 'ave_cong']
    df_processed[feat_scale_list] = scaler.fit_transform(df_processed[feat_scale_list])

    return df_processed


def plot_confusion(y_test, y_pred):
    """
    Plots a confusion matrix and prints out the accuracy, precision, recall, and F1 score.

    :param y_test: Testing data.
    :param y_pred: Predictions.
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f_score = f1_score(y_test, y_pred, average='weighted')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f_score}')

    # Plot a confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
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
