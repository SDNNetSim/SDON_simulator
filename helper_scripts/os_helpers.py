import os


def create_dir(file_path: str):
    """
    Create a directory at the specified file path if it doesn't already exist.

    :param file_path: The path to the file whose parent directory should be created.
    """
    if file_path is None:
        raise ValueError("File path cannot be None.")

    parent_dir_path = os.path.abspath(os.path.join(file_path, os.pardir))
    os.makedirs(parent_dir_path, exist_ok=True)

    last_child_dir_name = os.path.basename(file_path)
    last_child_dir_path = os.path.join(parent_dir_path, last_child_dir_name)
    os.makedirs(last_child_dir_path, exist_ok=True)
