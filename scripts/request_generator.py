import numpy as np


def generate(seed_no, nodes, holding_time_mean, inter_arrival_time_mean, req_no,
             slot_dict):
    """
    Generates every request with the necessary information inside each request.

    :param seed_no: The simulation seed number
    :type seed_no: int
    :param nodes: A list of nodes
    :type nodes: list
    :param holding_time_mean: The holding time mean
    :type holding_time_mean: int
    :param inter_arrival_time_mean: The arrival time mean
    :type inter_arrival_time_mean: int
    :param req_no: Number of requests to be created
    :type req_no: int
    :param slot_dict: A dictionary mapping bandwidths to slot numbers
    :return: Every request generated
    :rtype: dict
    """
    np.random.seed(seed_no)
    requests = {}
    current = 0
    counter_id = 0
    num_requests = len(requests)

    # TODO: Split number of requests length by these margins, make sure everything is included!
    #  (Correct number of requests)
    # Determines the ratio of requests allocated to each bandwidth
    bw_req_nums = {
        "10": {"max": num_requests / 3, "current": 0},
        "50": {"max": num_requests / 5, "current": 0},
        "400": {"max": num_requests / 2, "current": 0},
    }

    while num_requests < (req_no * 2):
        current = current + np.random.exponential(inter_arrival_time_mean)
        new_hold = current + np.random.exponential(holding_time_mean)

        src = nodes[np.random.randint(0, len(nodes))]
        des = nodes[np.random.randint(0, len(nodes))]

        while src == des:
            des = nodes[np.random.randint(0, len(nodes))]

        # TODO: Here
        bands_list = list(slot_dict.keys())
        chosen_band = bands_list[np.random.randint(0, len(bands_list))]  # pylint: disable=invalid-sequence-index
        slot_num = slot_dict[chosen_band]['DP-QPSK']

        if current not in requests and new_hold not in requests:
            requests.update({current: {
                "id": counter_id,
                "source": src,
                "destination": des,
                "arrive": current,
                "depart": new_hold,
                "request_type": "Arrival",
                "number_of_slot": slot_num,
                "start_slot_NO": None,
                "working_path": None,
                "protection_path": None

            }})
            requests.update({new_hold: {
                "id": counter_id,
                "source": src,
                "destination": des,
                "arrive": current,
                "depart": new_hold,
                "request_type": "Release",
                "number_of_slot": slot_num,
                "start_slot_NO": None,
                "working_path": None,
                "protection_path": None

            }})
            counter_id += 1
        else:
            continue
            # raise NotImplementedError('This line of code should not be reached.')

    return requests
