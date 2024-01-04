empty_props = {
    'iter_stats': dict(),
    'snapshots_dict': dict(),
    'cores_dict': dict(),
    'weights_dict': dict(),
    'mods_used_dict': dict(),
    'block_bw_dict': dict(),
    'block_reasons_dict': {'distance': None, 'congestion': None, 'xt_threshold': None},

    'sim_block_list': list(),
    'trans_list': list(),
    'hops_list': list(),
    'lengths_list': list(),
    'route_times_list': list(),
}

SNAP_KEYS_LIST = ['occupied_slots', 'guard_slots', 'active_requests', 'blocking_prob', 'num_segments']
