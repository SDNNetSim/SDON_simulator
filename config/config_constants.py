import ast

YUE_REQUIRED_OPTIONS = {
    'sim_type': str,
    'network': str,
    'holding_time': float,
    'arrival_rate': ast.literal_eval,
    'num_requests': int,
    'max_iters': int,
    'spectral_slots': int,
    'beta': float,
    'freq_per_slot': float,
    'cores_per_link': int,
    'const_link_weight': bool,
    'guard_slots': int,
    'max_segments': int,
    'dynamic_lps': bool,
    'allocation_method': str,
    'route_method': str,
    'request_distribution': ast.literal_eval,
}

ARASH_REQUIRED_OPTIONS = {

}
