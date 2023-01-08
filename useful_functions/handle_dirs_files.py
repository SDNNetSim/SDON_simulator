import os


def create_dir(fp=None):
    """
    Checks if a directory exists, if it doesn't create one.

    :param fp: The file path
    :type fp: str
    :return: None
    """
    if fp is None:
        raise ValueError(f'Expecting a valid file path, got: {fp}')

    # Iteratively check if the file path exists
    split_path = fp.split('/')
    curr_path = ''
    for i in range(len(split_path)):
        curr_path += split_path[i] + '/'
        if not os.path.exists(curr_path):
            os.mkdir(curr_path)
