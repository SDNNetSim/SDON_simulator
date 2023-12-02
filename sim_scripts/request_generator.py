import warnings

# Local application imports
from useful_functions.random_generation import set_seed, uniform_rv, exponential_rv


def generate(sim_type: str, seed: int, nodes: list, hold_time_mean: float, arr_rate_mean: float,
             num_reqs: int, mod_per_bw: dict, req_dist: dict):
    """
    Generates a dictionary of requests for a simulation, utilizing random number generation for arrival and departure times, source and destination nodes, and bandwidth selection. The resulting dictionary includes details such as ID, source, destination, arrival and departure times, request type, bandwidth, and modulation formats.

    :param sim_type: The simulation type.
    :type sim_type: str

    :param seed: Seed for random number generation.
    :type seed: int

    :param nodes: List of nodes in the network.
    :type nodes: list

    :param hold_time_mean: Mean hold time of a request.
    :type hold_time_mean: float

    :param arr_rate_mean: Mean arrival rate of requests.
    :type arr_rate_mean: float

    :param num_reqs: Number of requests to generate.
    :type num_reqs: int

    :param mod_per_bw: Dictionary with modulation formats for each bandwidth.
    :type mod_per_bw: dict

    :param req_dist: Dictionary with the distribution of requests among different bandwidths.
    :type req_dist: dict

    :return: A dictionary with the generated requests.
    :rtype: dict
    """
    # Initialize variables
    requests = {}
    current_time = 0
    counter_id = 1
    set_seed(seed=seed)
    warnings.warn('Request generation only works if the amount of requests can be evenly distributed.')

    bandwidth_counts = {bandwidth: int(req_dist[bandwidth] * num_reqs) for bandwidth in
                        mod_per_bw}
    bandwidth_list = list(mod_per_bw.keys())

    # Generate requests, multiply number of requests by two since we have arrival and departure types
    while len(requests) < (num_reqs * 2):
        current_time += exponential_rv(arr_rate_mean)

        if sim_type == 'arash':
            depart_time = current_time + exponential_rv(1 / hold_time_mean)
        else:
            depart_time = current_time + exponential_rv(hold_time_mean)

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
                "mod_formats": mod_per_bw[chosen_bandwidth],
            }})
            requests.update({depart_time: {
                "id": counter_id,
                "source": source,
                "destination": dest,
                "arrive": current_time,
                "depart": depart_time,
                "request_type": "release",
                "bandwidth": chosen_bandwidth,
                "mod_formats": mod_per_bw[chosen_bandwidth],
            }})
            counter_id += 1
        else:
            bandwidth_counts[chosen_bandwidth] += 1

    return requests
