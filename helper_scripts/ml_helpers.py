import pandas as pd


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


def process_data(input_df: pd.DataFrame):
    input_df['mod_format'] = input_df['mod_format'].str.replace('-', '')
    df_processed = pd.get_dummies(input_df, columns=['bandwidth', 'mod_format'])

    for col in df_processed.columns:
        if df_processed[col].dtype == bool:
            df_processed[col] = df_processed[col].astype(int)

    return df_processed
