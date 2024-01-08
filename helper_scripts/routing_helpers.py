import numpy as np

from helper_scripts.sim_helpers import find_path_len, get_path_mod
from sim_scripts.routing import Routing


# TODO: Can eventually include this in routing and delete this file or put other things in it
# TODO: Will eventually include routing in sdn_controller
# TODO: Standardize return format
def get_route(engine_props: dict, sdn_props: dict, ai_obj: object):
    routing_obj = Routing(engine_props=engine_props, sdn_props=sdn_props)

    # TODO: Run all of these to make sure they work and have the same results as prior commits
    # TODO: Using variables multiple times? declare here?
    if engine_props['route_method'] == 'nli_aware':
        slots_needed = engine_props['mod_per_bw'][sdn_props['chosen_bw']]['QPSK']['slots_needed']
        routing_obj.slots_needed = slots_needed
        routing_obj.beta = engine_props['beta']
        resp = routing_obj.nli_aware()
    elif engine_props['route_method'] == 'xt_aware':
        resp = routing_obj.xt_aware(beta=engine_props['beta'], xt_type=engine_props['xt_type'])
    elif engine_props['route_method'] == 'least_congested':
        resp = routing_obj.find_least_cong_path()
    elif engine_props['route_method'] == 'shortest_path':
        resp = routing_obj.find_least_weight_path(weight='length')
    elif engine_props['route_method'] == 'k_shortest_path':
        resp = routing_obj.k_shortest_path(k_paths=engine_props['k_paths'])
    elif engine_props['route_method'] == 'ai':
        # Used for routing related to artificial intelligence
        # TODO: Just pass properties here when you get to ai_obj
        selected_path, selected_core = ai_obj.route(source=int(sdn_props['source']),
                                                    destination=int(sdn_props['destination']),
                                                    net_spec_db=sdn_props['net_spec_db'],
                                                    chosen_bw=sdn_props['chosen_bw'],
                                                    guard_slots=engine_props['guard_slots'])

        # A path could not be found, assign None to path modulation
        if not selected_path:
            resp = [selected_path], [False], [False], [False]
        else:
            path_len = find_path_len(path=selected_path, topology=sdn_props['topology'])
            path_mod = [get_path_mod(mod_formats=engine_props['mod_per_bw'][sdn_props['chosen_bw']], path_len=path_len)]
            resp = [selected_path], [selected_core], [path_mod], [path_len]
    else:
        raise NotImplementedError(f"Routing method not recognized, got: {engine_props['route_method']}.")

    # TODO: Return routing props here
    return resp


def get_simulated_link(slots_needed: int, guard_slots: int, slots_per_core: int):
    # Simulate a fully congested link
    simulated_link = np.zeros(slots_per_core)

    # Add to the step to account for the guard band
    for i in range(0, len(simulated_link), slots_needed + guard_slots):
        value_to_set = i // slots_needed + 1
        simulated_link[i:i + slots_needed + 2] = value_to_set

    # Add guard-bands
    simulated_link[slots_needed::slots_needed + guard_slots] *= -1

    # Free the middle-most channel with respect to the number of slots needed
    center_index = len(simulated_link) // 2
    if slots_needed % 2 == 0:
        start_index = center_index - slots_needed // 2
        end_idx = center_index + slots_needed // 2
    else:
        start_index = center_index - slots_needed // 2
        end_idx = center_index + slots_needed // 2 + 1

    simulated_link[start_index:end_idx] = 0

    return simulated_link
