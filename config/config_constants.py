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
    'holding_time': float,
    'arrival_rate': ast.literal_eval,
    'num_requests': int,
    'max_iters': int,
    'spectral_slots': int,
    'freq_per_slot': float,
    'cores_per_link': int,
    'const_link_weight': str_to_bool,
    'guard_slots': int,
    'max_segments': int,
    'dynamic_lps': str_to_bool,
    'allocation_method': str,
    'route_method': str,
    'request_distribution': ast.literal_eval,
}

ARASH_REQUIRED_OPTIONS = {
    'sim_type': str,
    'network': str,
    'holding_time': float,
    'num_requests': int,
    'max_iters': int,
    'spectral_slots': int,
    'freq_per_slot': float,
    'cores_per_link': int,
    'const_link_weight': str_to_bool,
    'guard_slots': int,
    'max_segments': int,
    'dynamic_lps': str_to_bool,
    'allocation_method': str,
    'route_method': str,
    'request_distribution': ast.literal_eval,
    'beta': float,
}

OTHER_OPTIONS = {
    'ai_algorithm': str,
    'is_training': str_to_bool,
    'seeds': list,
    'beta': float,
    'train_file': str,
}
