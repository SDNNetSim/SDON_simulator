# import random
import numpy as np
import math


def Generate(seed_no, nodes, holding_time_mean, inter_arrival_time_mean, req_no, slot_list):
    np.random.seed(seed_no)
    requests = {}
    current = 0
    for i in range(0, req_no):
        current = current + np.random.normal(loc=inter_arrival_time_mean)
        new_hold = current + np.random.normal(loc=holding_time_mean)
        # TODO: Should this be fed a seed?
        np.random.RandomState.randint
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
            print("Check")
            raise "rep"
    return requests
