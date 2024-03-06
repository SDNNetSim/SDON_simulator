import ast


def str_to_bool(string: str):
    """
    Convert any giving string to a boolean.

    :param string: The input string.
    :return: True or False
    :rtype: bool
    """
    return string.lower() in ['true', 'yes', '1']


YUE_REQUIRED_OPTIONS = {
    'general_settings': {
        'sim_type': str,
        'holding_time': float,
        'arrival_rate': ast.literal_eval,
        'thread_erlangs': str_to_bool,
        'guard_slots': int,
        'num_requests': int,
        'request_distribution': ast.literal_eval,
        'max_iters': int,
        'max_segments': int,
        'dynamic_lps': str_to_bool,
        'allocation_method': str,
        'route_method': str,
        'save_snapshots': str_to_bool,
        'snapshot_step': int,
        'print_step': int,
    },
    'topology_settings': {
        'network': str,
        'spectral_slots': int,
        'bw_per_slot': float,
        'cores_per_link': int,
        'const_link_weight': str_to_bool,
    },
    'file_settings': {
        'file_type': str,
    },
}

ARASH_REQUIRED_OPTIONS = {
    'general_settings': {
        'sim_type': str,
        'holding_time': float,
        'erlangs': ast.literal_eval,
        'thread_erlangs': str_to_bool,
        'guard_slots': int,
        'num_requests': int,
        'request_distribution': ast.literal_eval,
        'max_iters': int,
        'max_segments': int,
        'dynamic_lps': str_to_bool,
        'route_method': str,
        'allocation_method': str,
        'save_snapshots': str_to_bool,
        'snapshot_step': int,
        'print_step': int,
    },
    'topology_settings': {
        'network': str,
        'spectral_slots': int,
        'bw_per_slot': float,
        'cores_per_link': int,
        'const_link_weight': str_to_bool,
    },
    'snr_settings': {
        'requested_xt': ast.literal_eval,
        'xt_noise': str_to_bool,
        'theta': float,
        'egn_model': str_to_bool,
        'phi': ast.literal_eval,
        'snr_type': str,
        'xt_type': str,
        'beta': float,
        'input_power': float,
    },
    'file_settings': {
        'file_type': str,
    },
}

OTHER_OPTIONS = {
    'general_settings': {
        'seeds': list,
        'k_paths': int,
        'filter_mods': bool,
        'snapshot_step': int,
        'print_step': int,
    },
    'topology_settings': {
        'bi_directional': str_to_bool,
    },
    'snr_settings': {
        'snr_type': str,
        'xt_type': str,
        'theta': float,
        'beta': float,
        'input_power': float,
        'egn_model': str_to_bool,
        'phi': ast.literal_eval,
        'xt_noise': str_to_bool,
        'requested_xt': ast.literal_eval,
    },
    'ai_settings': {
        'ai_algorithm': str,
        'learn_rate': float,
        'discount_factor': float,
        'epsilon_start': float,
        'epsilon_end': float,
        'is_training': str,
    },
    'file_settings': {
    },
}

COMMAND_LINE_PARAMS = [
    ['ai_algorithm', str, ''],
    ['epsilon_start', float, ''],
    ['epsilon_end', float, ''],
    ['learn_rate', float, ''],
    ['discount_factor', float, ''],
    ['is_training', float, ''],
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

    # TODO: Modify? I'm not sure if my model's params are actually updating
    # TODO: I believe this are meant to input into the script
    # TODO: Add all args
    ['algo', str, ''],
    ['env', str, ''],
    ['eval-freq', str, ''],
    ['save-freq', str, ''],
    ['n-trials', int, ''],
    ['n-jobs', int, ''],
    ['n', int, ''],
    ['sampler', str, ''],
    ['pruner', str, ''],
    ['eval-episodes', int, ''],
    ['n-startup-trials', int, ''],
    ['train-freq', int, ''],
    ['num-threads', int, ''],
    ['max-total-trials', int, ''],
    ['n-timesteps', int, ''],
]
