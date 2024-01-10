from helper_scripts.random_helpers import set_seed, get_uniform_rv, get_exponential_rv


def get_requests(seed: int, engine_props: dict):
    """
    Generates requests for a single simulation.

    :param seed: Seed for random generation.
    :param engine_props: Properties from the engine class.
    :return: The generated requests and request information.
    :rtype: dict
    """
    requests_dict = {}
    current_time = 0
    request_id = 1

    nodes = list(engine_props['topology_info']['nodes'].keys())
    set_seed(seed=seed)

    # Create request distribution
    bandwidth_counts = {bandwidth: int(engine_props['request_distribution'][bandwidth] * engine_props['num_requests'])
                        for bandwidth in engine_props['mod_per_bw']}
    bandwidth_list = list(engine_props['mod_per_bw'].keys())

    # Generate requests, multiply the number of requests by two since we have arrival and departure types
    while len(requests_dict) < (engine_props['num_requests'] * 2):
        current_time += get_exponential_rv(engine_props['arrival_rate'])

        if engine_props['sim_type'] == 'arash':
            depart_time = current_time + get_exponential_rv(1 / engine_props['holding_time'])
        else:
            depart_time = current_time + get_exponential_rv(engine_props['holding_time'])

        source = nodes[get_uniform_rv(len(nodes))]
        dest = nodes[get_uniform_rv(len(nodes))]

        while dest == source:
            dest = nodes[get_uniform_rv(len(nodes))]

        while True:
            chosen_bandwidth = bandwidth_list[get_uniform_rv(len(bandwidth_list))]
            if bandwidth_counts[chosen_bandwidth] > 0:
                bandwidth_counts[chosen_bandwidth] -= 1
                break

        if current_time not in requests_dict and depart_time not in requests_dict:
            requests_dict.update({current_time: {
                "req_id": request_id,
                "source": source,
                "destination": dest,
                "arrive": current_time,
                "depart": depart_time,
                "request_type": "arrival",
                "bandwidth": chosen_bandwidth,
                "mod_formats": engine_props['mod_per_bw'][chosen_bandwidth],
            }})
            requests_dict.update({depart_time: {
                "req_id": request_id,
                "source": source,
                "destination": dest,
                "arrive": current_time,
                "depart": depart_time,
                "request_type": "release",
                "bandwidth": chosen_bandwidth,
                "mod_formats": engine_props['mod_per_bw'][chosen_bandwidth],
            }})
            request_id += 1
        # Bandwidth was not chosen due to either arrival or depart time already existing, add back to distribution
        else:
            bandwidth_counts[chosen_bandwidth] += 1

    return requests_dict
