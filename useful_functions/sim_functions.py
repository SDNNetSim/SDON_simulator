def get_path_mod(mod_obj, path_len):
    """
    Given an object of modulation formats and maximum lengths, choose the one that satisfies the requirements.

    :param mod_obj: The modulation object, holds needed information for maximum reach
    :type mod_obj: dict
    :param path_len: The length of the path to be taken
    :type path_len: int
    :return: The chosen modulation format, or false
    """
    if mod_obj['QPSK']['max_length'] >= path_len > mod_obj['16-QAM']['max_length']:
        resp = 'QPSK'
    elif mod_obj['16-QAM']['max_length'] >= path_len > mod_obj['64-QAM']['max_length']:
        resp = '16-QAM'
    elif mod_obj['64-QAM']['max_length'] >= path_len:
        resp = '64-QAM'
    else:
        return False

    return resp


def sort_dict_keys(obj):
    """
    Given a dictionary with key value pairs, sort the dictionary by keys.

    :param obj: The dictionary
    :return: The sorted dictionary
    """
    keys_lst = [int(key) for key in obj.keys()]
    keys_lst.sort(reverse=True)
    sorted_obj = {str(i): obj[str(i)] for i in keys_lst}

    return sorted_obj


def find_path_len(path, topology):
    """
    Finds the length of the given path in the physical topology.

    :param path: The path a request takes from node 'a' to node 'b'
    :type path: list
    :param topology: The physical topology of the simulation
    :type: graph
    :return: The length of the path
    :rtype: int
    """
    path_len = 0
    for i in range(len(path) - 1):
        path_len += topology[path[i]][path[i + 1]]['length']

    return path_len
