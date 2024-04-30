import os

empty_props = {
    'sims_info_dict': None,
    'plot_dict': None,
    'output_dir': os.path.join('..', 'data', 'output'),
    'input_dir': os.path.join('..', 'data', 'input'),
    'sim_num': None,
    'erlang_dict': None,
    'num_requests': None,
    'num_cores': None,

    'color_list': ['#024de3', '#00b300', 'orange', '#6804cc', '#e30220'],
    'style_list': ['solid', 'dashed', 'dotted', 'dashdot'],
    'marker_list': ['o', '^', 's', 'x'],
    'x_tick_list': [50, 100, 200, 300, 400, 500, 600, 700],
    'title_names': None,
}

empty_plot_dict = {
    'erlang_list': [],
    'blocking_list': [],
    'lengths_list': [],
    'hops_list': [],
    'occ_slot_matrix': [],
    'active_req_matrix': [],
    'block_req_matrix': [],
    'req_num_list': [],
    'times_list': [],
    'modulations_dict': dict(),
    'dist_block_list': [],
    'cong_block_list': [],
    'holding_time': None,
    'cores_per_link': None,
    'spectral_slots': None,
    'learn_rate': None,
    'discount_factor': None,

    'block_per_iter': [],
    'sum_rewards_list': [],
    'sum_errors_list': [],
    'epsilon_list': [],
}
