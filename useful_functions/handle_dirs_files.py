# Standard library imports
import os


def create_dir(file_path: str):
    """
    Create a directory at the specified file path if it doesn't already exist.

    :param file_path: The path to the file whose parent directory should be created.
    :type file_path: str

    :return: None
    """
    if file_path is None:
        raise ValueError("File path cannot be None.")

    # Get the absolute path of the parent directory and create it if it doesn't exist
    parent_dir_path = os.path.abspath(os.path.join(file_path, os.pardir))
    os.makedirs(parent_dir_path, exist_ok=True)

    # Get the last child directory name and create it if it doesn't exist
    last_child_dir_name = os.path.basename(file_path)
    last_child_dir_path = os.path.join(parent_dir_path, last_child_dir_name)
    os.makedirs(last_child_dir_path, exist_ok=True)
