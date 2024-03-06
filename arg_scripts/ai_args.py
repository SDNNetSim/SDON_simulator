empty_q_props = {
    'epsilon': None,
    'epsilon_start': None,
    'epsilon_end': None,
    'epsilon_list': list(),
    'is_training': None,

    'rewards_dict': {
        'routes_dict': {'average': [], 'min': [], 'max': [], 'rewards': {}},
        'cores_dict': {'average': [], 'min': [], 'max': [], 'rewards': {}}
    },
    'errors_dict': {
        'routes_dict': {'average': [], 'min': [], 'max': [], 'errors': {}},
        'cores_dict': {'average': [], 'min': [], 'max': [], 'errors': {}}
    },
    'sum_rewards_dict': dict(),
    'sum_errors_dict': dict(),

    'routes_matrix': None,
    'cores_matrix': None,
    'num_nodes': None,

    'save_params_dict': {
        'q_params_list': ['rewards_dict', 'errors_dict', 'epsilon_list', 'sum_rewards_dict', 'sum_errors_dict'],
        'engine_params_list': ['epsilon_start', 'epsilon_end', 'max_iters', 'learn_rate', 'discount_factor']
    }
}

empty_dqn_props = {
    'net_spec_dict': dict(),
    'arrival_list': list(),
    'depart_list': list(),
    'arrival_count': 0,

    'engine_props': None,
    'reqs_dict': None,
    'mock_sdn_dict': dict,
}
