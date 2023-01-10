def get_path_mod(mod_obj, path_len):
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
    keys_lst = [int(key) for key in obj.keys()]
    keys_lst.sort(reverse=True)
    sorted_obj = {str(i): obj[str(i)] for i in keys_lst}

    return sorted_obj


def find_path_len(path, topology):
    path_len = 0
    for i in range(len(path) - 1):
        path_len += topology[path[i]][path[i + 1]]['length']

    return path_len
