import math

from useful_functions.random_generation import *


def generate(seed_no, nodes, mu, lam, num_requests, bw_dict):
    """
    Generate all the requests for the simulation.

    :param seed_no: The seed
    :type seed_no: int
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
    :return: Every request generated
    :rtype: dict
    """
    requests = {}
    current_time = 0
    counter_id = 1

    set_seed(seed_no=seed_no)

    # Bandwidth ratio generation
    # TODO: This should be moved to run_sim
    bw_ratio_one = 0.0
    bw_ratio_two = 0.5
    bw_ratio_three = 0.5

    # Number of requests allocated for each bandwidth
    # TODO: This could potentially not equal the number of requests we think
    bw_one_req = bw_ratio_one * num_requests
    bw_two_req = bw_ratio_two * num_requests
    bw_three_req = bw_ratio_three * num_requests

    # Monitor the number of requests allocated for each bandwidth
    bands_dict = {'50': bw_one_req, '100': bw_two_req, '400': bw_three_req}
    # List of all possible bandwidths
    bands_list = list(bw_dict.keys())

    # Multiplied by two, to account for arrival and departure requests
    while len(requests) < (num_requests * 2):
        current_time = current_time + (math.ceil(exponential_rv(lam) * 1000) / 1000)
        depart_time = current_time + 9000000000000 #(math.ceil(exponential_rv(mu) * 1000) / 1000)

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
            else:
                continue
        chosen_bw = '50'
        if counter_id == 251:
            chosen_bw = '400'
        if current_time not in requests and depart_time not in requests:
            requests.update({current_time: {
                "id": counter_id,
                "source": '0',#src,
                "destination": '1', #des,
                "arrive": current_time,
                "depart": depart_time,
                "request_type": "arrival",
                "bandwidth": chosen_bw,
                "mod_formats": bw_dict[chosen_bw],
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
                "mod_formats": bw_dict[chosen_bw],
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
