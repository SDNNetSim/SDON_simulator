import ast


def str_to_bool(string):
    """
    Convert any giving string to a boolean.

    :param string: The input string.
    :type string: str

    :return: True or False
    :rtype: bool
    """
    return string.lower() in ['true', 'yes', '1']


YUE_REQUIRED_OPTIONS = {
    'sim_type': str,
    'network': str,
    'warnings': str_to_bool,
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
}

ARASH_REQUIRED_OPTIONS = {
    'sim_type': str,
    'network': str,
    'warnings': str_to_bool,
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
    'xt_type': str,
}

OTHER_OPTIONS = {
    'ai_algorithm': str,
    'ai_arguments': ast.literal_eval,
    'seeds': list,
    'beta': float,
    'train_file': str,
    'check_snr': str,
    'input_power': float,
    'egn_model': str_to_bool,
    'phi': ast.literal_eval,
    'bi_directional': str_to_bool,
    'xt_noise': str_to_bool,
    'requested_xt': ast.literal_eval,
    'k_paths': int,
    'xt_type': str,
}
