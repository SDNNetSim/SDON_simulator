import math
import numpy as np


# TODO: Pay attention to system time, it's different than arrival in his code


# TODO: Move these to 'useful functions'
def universal_rv():
    return float(np.random.randint(0, 2147483647)) / float(2147483647)


# TODO: Ensure that passing the list length minus one is okay
def uniform_rv(scale_param):
    return int(universal_rv() * scale_param)


def exponential_rv(scale_param):
    # np.log is the natural logarithm
    return ((-1.0) / float(scale_param)) * np.log(universal_rv())


def generate(seed_no, nodes, holding_time_mean, inter_arrival_time_mean, num_requests,
             slot_dict):
    np.random.seed(seed_no)
    requests = {}
    current_time = 0
    counter_id = 0

    # We want to create requests in a 3:5:2 fashion
    # TODO: Move this to the engine eventually
    bw_ratio_one = 0.3
    bw_ratio_two = 0.5
    bw_ratio_three = 0.2

    # Number of requests allocated for each bandwidth
    # TODO: This could potentially not equal the number of requests we think
    # Multiplied by two, to account for arrival and departure requests
    bw_one_req = bw_ratio_one * num_requests
    bw_two_req = bw_ratio_two * num_requests
    bw_three_req = bw_ratio_three * num_requests

    # TODO: (Question for Yue) His code uses 40 Gbps AND, not listed in the table?
    # Monitor the number of requests allocated for each bandwidth
    bands_dict = {'50': bw_one_req, '100': bw_two_req, '400': bw_three_req}
    # List of all possible bandwidths
    bands_list = list(slot_dict.keys())
    while len(requests) < (num_requests * 2):
        # TODO: Yue turns his into integers?
        # TODO: Try with ints
        # We should probably start at zero
        current_time = current_time + (math.ceil(exponential_rv(inter_arrival_time_mean) * 1000) / 1000)
        depart_time = current_time + (math.ceil(exponential_rv(holding_time_mean) * 1000) / 1000)

        # We never want our node to equal the length, we start from index 0 in a list! (Node numbers are all minus 1)
        # TODO: Ensure all nodes are being utilized
        src = nodes[uniform_rv(len(nodes))]
        des = nodes[uniform_rv(len(nodes))]

        while src == des:
            des = nodes[uniform_rv(len(nodes))]

        while True:
            # TODO: He sets a datasize for each (derives occupied slots based on this)
            # TODO: Ensure the correct amount of bws are allocated
            chosen_bw = bands_list[uniform_rv(len(bands_list))]  # pylint: disable=invalid-sequence-index
            if bands_dict[chosen_bw] > 0:
                bands_dict[chosen_bw] -= 1
                break
            else:
                continue

        # TODO: Requests should be allowed to have the same arrival time?
        if current_time not in requests and depart_time not in requests:
            requests.update({current_time: {
                "id": counter_id,
                "source": src,
                "destination": des,
                "arrive": current_time,
                "depart": depart_time,
                "request_type": "Arrival",
                "bandwidth": chosen_bw,
                "mod_formats": slot_dict[chosen_bw],
                "start_slot_NO": None,
                "working_path": None,
                "protection_path": None
            }})
            requests.update({depart_time: {
                "id": counter_id,
                "source": src,
                "destination": des,
                "arrive": current_time,
                "depart": depart_time,
                "request_type": "Release",
                "bandwidth": chosen_bw,
                "mod_formats": slot_dict[chosen_bw],
                "start_slot_NO": None,
                "working_path": None,
                "protection_path": None
            }})
            counter_id += 1
        else:
            bands_dict[chosen_bw] += 1
            continue

    return requests
