empty_q_props = {
    'epsilon': None,
    'epsilon_start': None,
    'epsilon_end': None,
    'epsilon_list': None,
    'is_training': None,

    'rewards_dict': None,
    'errors_dict': None,
    'sum_rewards': None,

    'routes_matrix': None,
    'cores_matrix': None,
    'num_nodes': None,

    'save_params_dict': {
        'q_params_list': ['rewards_dict', 'errors_dict', 'epsilon_list', 'sum_rewards'],
        'engine_params_list': ['epsilon_start', 'epsilon_end', 'max_iters', 'learn_rate', 'discount_factor']
    }
}
