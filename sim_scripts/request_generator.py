import warnings

# Local application imports
from useful_functions.random_generation import set_seed, uniform_rv, exponential_rv


def generate(seed: int, engine_props: dict):
    """
    Generates requests for a simulation.

    :param seed: Seed for random number generation.
    :param engine_props: Properties from the engine class.
    :return: A dictionary with the generated requests and request information.
    :rtype: dict
    """
    requests = {}
    nodes = list(engine_props['topology_info']['nodes'].keys())
    current_time = 0
    counter_id = 1
    set_seed(seed=seed)
    # Fixme
    warnings.warn('Request generation only works if the amount of requests can be evenly distributed.')

    bandwidth_counts = {bandwidth: int(engine_props['request_distribution'][bandwidth] * engine_props['num_requests'])
                        for bandwidth in engine_props['mod_per_bw']}
    bandwidth_list = list(engine_props['mod_per_bw'].keys())

    # Generate requests, multiply number of requests by two since we have arrival and departure types
    while len(requests) < (engine_props['num_requests'] * 2):
        current_time += exponential_rv(engine_props['arrival_rate'])

        if engine_props['sim_type'] == 'arash':
            depart_time = current_time + exponential_rv(1 / engine_props['holding_time'])
        else:
            depart_time = current_time + exponential_rv(engine_props['holding_time'])

        source = nodes[uniform_rv(len(nodes))]
        dest = nodes[uniform_rv(len(nodes))]

        while dest == source:
            dest = nodes[uniform_rv(len(nodes))]

        while True:
            chosen_bandwidth = bandwidth_list[uniform_rv(len(bandwidth_list))]
            if bandwidth_counts[chosen_bandwidth] > 0:
                bandwidth_counts[chosen_bandwidth] -= 1
                break

        if current_time not in requests and depart_time not in requests:
            requests.update({current_time: {
                "id": counter_id,
                "source": source,
                "destination": dest,
                "arrive": current_time,
                "depart": depart_time,
                "request_type": "arrival",
                "bandwidth": chosen_bandwidth,
                "mod_formats": engine_props['mod_per_bw'][chosen_bandwidth],
            }})
            requests.update({depart_time: {
                "id": counter_id,
                "source": source,
                "destination": dest,
                "arrive": current_time,
                "depart": depart_time,
                "request_type": "release",
                "bandwidth": chosen_bandwidth,
                "mod_formats": engine_props['mod_per_bw'][chosen_bandwidth],
            }})
            counter_id += 1
        else:
            bandwidth_counts[chosen_bandwidth] += 1

    return requests
