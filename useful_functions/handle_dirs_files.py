import os


def create_dir(file_path=None):
    """
    Checks if a directory exists, if it doesn't create one.

    :param file_path: The file path
    :type file_path: str
    :return: None
    """
    if file_path is None:
        raise ValueError(f'Expecting a valid file path, got: {file_path}')

    # Iteratively check if the file path exists
    split_path = file_path.split('/')
    curr_path = ''
    for i in range(len(split_path)):  # pylint: disable=consider-using-enumerate
        curr_path += split_path[i] + '/'
        if not os.path.exists(curr_path):
            os.mkdir(curr_path)
