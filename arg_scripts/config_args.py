import ast


def str_to_bool(string):
    """
    Convert any giving string to a boolean.

    :param string: The input string.
    :return: True or False
    :rtype: bool
    """
    return string.lower() in ['true', 'yes', '1']


YUE_REQUIRED_OPTIONS = {
    'sim_type': str,
    'network': str,
    'bw_per_slot': float,
    'thread_erlangs': str_to_bool,
    'holding_time': float,
    'arrival_rate': ast.literal_eval,
    'num_requests': int,
    'max_iters': int,
    'spectral_slots': int,
    'cores_per_link': int,
    'const_link_weight': str_to_bool,
    'guard_slots': int,
    'max_segments': int,
    'dynamic_lps': str_to_bool,
    'allocation_method': str,
    'route_method': str,
    'request_distribution': ast.literal_eval,
    'save_snapshots': str_to_bool,
    'snapshot_step': int,
    'print_step': int,
    'file_type': str,
}

ARASH_REQUIRED_OPTIONS = {
    'sim_type': str,
    'network': str,
    'holding_time': float,
    'erlangs': ast.literal_eval,
    'thread_erlangs': str_to_bool,
    'num_requests': int,
    'max_iters': int,
    'spectral_slots': int,
    'bw_per_slot': float,
    'cores_per_link': int,
    'const_link_weight': str_to_bool,
    'guard_slots': int,
    'max_segments': int,
    'dynamic_lps': str_to_bool,
    'allocation_method': str,
    'route_method': str,
    'request_distribution': ast.literal_eval,
    'beta': float,
    'save_snapshots': str_to_bool,
    'snapshot_step': int,
    'print_step': int,
    'file_type': str,
    'xt_type': str,
}

OTHER_OPTIONS = {
    'ai_algorithm': str,
    'policy': str,
    'seeds': list,
    'beta': float,
    'train_file': str,
    'snr_type': str,
    'input_power': float,
    'egn_model': str_to_bool,
    'phi': ast.literal_eval,
    'bi_directional': str_to_bool,
    'xt_noise': str_to_bool,
    'requested_xt': ast.literal_eval,
    'k_paths': int,
    'xt_type': str,
    'snapshot_step': int,
    'print_step': int,
    'file_type': str,
    'theta': float,
    'filter_mods': bool,
}

COMMAND_LINE_PARAMS = [
    ['ai_algorithm', str, ''],
    ['policy', str, ''],
    ['seeds', list, ''],
    ['beta', float, ''],
    ['train_file', str, ''],
    ['snr_type', str, ''],
    ['input_power', float, ''],
    ['egn_model', bool, ''],
    ['phi', dict, ''],
    ['bi_directional', bool, ''],
    ['xt_noise', bool, ''],
    ['requested_xt', dict, ''],
    ['k_paths', int, ''],
    ['sim_type', str, ''],
    ['network', str, ''],
    ['holding_time', float, ''],
    ['erlangs', dict, ''],
    ['thread_erlangs', bool, ''],
    ['num_requests', int, ''],
    ['max_iters', int, ''],
    ['spectral_slots', int, ''],
    ['bw_per_slot', float, ''],
    ['cores_per_link', int, ''],
    ['const_link_weight', bool, ''],
    ['guard_slots', int, ''],
    ['max_segments', int, ''],
    ['dynamic_lps', bool, ''],
    ['allocation_method', str, ''],
    ['route_method', str, ''],
    ['request_distribution', dict, ''],
    ['arrival_rate', dict, ''],
    ['save_snapshots', bool, ''],
    ['xt_type', str, ''],
    ['snapshot_step', int, ''],
    ['print_step', int, ''],
    ['file_type', str, ''],
    ['theta', float, ''],
    ['filter_mods', bool, ''],
]
