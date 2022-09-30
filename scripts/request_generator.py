import numpy as np


def generate(seed_no, nodes, holding_time_mean, inter_arrival_time_mean, req_no,
             slot_list):
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
    :param slot_list: The slot list
    :return: Every request generated
    :rtype: dict
    """
    np.random.seed(seed_no)
    requests = {}
    current = 0

    for i in range(0, req_no):
        current = current + np.random.normal(loc=inter_arrival_time_mean)
        new_hold = current + np.random.normal(loc=holding_time_mean)
        src = np.random.choice(nodes, size=1)
        des = np.random.choice(nodes, size=1)

        while src == des:
            des = np.random.choice(nodes, size=1)
        slot_no = np.random.choice(slot_list, size=1)

        if current not in requests and new_hold not in requests:
            requests.update({current: {
                "id": i,
                "source": src,
                "destination": des,
                "arrive": current,
                "depart": new_hold,
                "request_type": "Arrival",
                "number_of_slot": slot_no,
                "start_slot_NO": None,
                "working_path": None,
                "protection_path": None

            }})
            requests.update({new_hold: {
                "id": i,
                "source": src,
                "destination": des,
                "arrive": current,
                "depart": new_hold,
                "request_type": "Release",
                "number_of_slot": slot_no,
                "start_slot_NO": None,
                "working_path": None,
                "protection_path": None

            }})
        else:
            # TODO: Shouldn't we raise a class here?
            raise "rep"  # pylint: disable=raising-bad-type

    return requests
