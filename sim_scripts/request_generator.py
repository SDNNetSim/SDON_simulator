from useful_functions.random_generation import set_seed, uniform_rv, exponential_rv


def generate(seed, nodes, hold_time_mean, arr_rate_mean, num_reqs, mod_per_bw, req_dist):
    """
    Generate all the requests for the simulation.

    :param seed: The seed
    :type seed: int
    :param nodes: All nodes in the network
    :type nodes: list
    :param mu: The holding time mean
    :type mu: float
    :param lam: The inter-arrival time mean
    :type lam: float
    :param num_requests: Amount of requests
    :type num_requests: int
    :param bw_dict: Contains information for each bandwidth and modulation
    :type bw_dict: dict
    :param req_dist: The distribution of requests we'd like to generate
    :type req_dist: dict
    :return: Every request generated
    :rtype: dict
    """
    requests = {}
    current_time = 0
    # We must start at 1, as we allocate spectral slots with the request number and '0' is considered a free spectral
    # slot
    counter_id = 1

    set_seed(seed=seed)

    # Number of requests allocated for each bandwidth
    # TODO: This could potentially not equal the number of requests we think
    bw_one_req = req_dist['25'] * num_reqs
    bw_two_req = req_dist['50'] * num_reqs
    bw_three_req = req_dist['100'] * num_reqs
    bw_four_req = req_dist['200'] * num_reqs
    bw_five_req = req_dist['400'] * num_reqs

    # Monitor the number of requests allocated for each bandwidth
    bands_dict = {'25': bw_one_req, '50': bw_two_req, '100': bw_three_req, '200': bw_four_req, '400': bw_five_req}
    # List of all possible bandwidths
    bands_list = list(mod_per_bw.keys())

    # Multiplied by two, to account for arrival and departure requests
    while len(requests) < (num_reqs * 2):
        current_time = current_time + exponential_rv(arr_rate_mean)
        depart_time = current_time + exponential_rv(hold_time_mean)

        # We never want our node to equal the length, we start from index 0 in a list! (Node numbers are all minus 1)
        src = nodes[uniform_rv(len(nodes))]
        des = nodes[uniform_rv(len(nodes))]

        while src == des:
            des = nodes[uniform_rv(len(nodes))]

        while True:
            chosen_bw = bands_list[uniform_rv(len(bands_list))]
            if bands_dict[chosen_bw] > 0:
                bands_dict[chosen_bw] -= 1
                break

        if current_time not in requests and depart_time not in requests:
            requests.update({current_time: {
                "id": counter_id,
                "source": src,
                "destination": des,
                "arrive": current_time,
                "depart": depart_time,
                "request_type": "arrival",
                "bandwidth": chosen_bw,
                "mod_formats": mod_per_bw[chosen_bw],
                "start_slot_no": None,
                "working_path": None,
                "protection_path": None
            }})
            requests.update({depart_time: {
                "id": counter_id,
                "source": src,
                "destination": des,
                "arrive": current_time,
                "depart": depart_time,
                "request_type": "release",
                "bandwidth": chosen_bw,
                "mod_formats": mod_per_bw[chosen_bw],
                "start_slot_no": None,
                "working_path": None,
                "protection_path": None
            }})
            counter_id += 1
        # Bandwidth wasn't chosen due to time conflict, request not allocated (previously subtracted)
        else:
            bands_dict[chosen_bw] += 1
            continue

    return requests
