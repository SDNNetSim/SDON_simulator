# Standard imports
import copy

# Third party imports
import networkx as nx
import numpy as np

# Project imports
from Request_generator import Generate
from load_input import load_input
from SDN_Controller import controller_main

blocking = {}
sim_input = load_input()


def create_PT(pt_input):
    """
    Creates the physical topology for the simulation. Adds nodes to the graph, updates the spectrum
    database with the source, destination, and slots per link, and adds the designated source and
    destination to the graph as edges.

    :param pt_input: Desired information to run the simulation
    :type pt_input: dict
    :return: Graph and spectrum database
    :rtype: class and dict
    """
    G = nx.Graph()
    spectrum_DB = {}

    for node in pt_input['physical_topology']['nodes']:
        G.add_node(node)

    for link_no in pt_input['physical_topology']['links']:
        spectrum_DB.update({(pt_input['physical_topology']['links'][link_no]['source'],
                             pt_input['physical_topology']['links'][link_no]['destination']):
                            np.zeros(pt_input['number_of_slot_per_lisnk'])})
        G.add_edge(pt_input['physical_topology']['links'][link_no]['source'],
                   pt_input['physical_topology']['links'][link_no]['destination'],
                   length=pt_input['physical_topology']['links'][link_no]['length'])

    return G, spectrum_DB


def main():
    """
    Executes the simulation.
    """
    for i in range(sim_input['NO_iteration']):
        blocking_iter = 0
        requests_status = {}
        pt_res = create_PT(sim_input)
        network_spectrum_DB = copy.deepcopy(pt_res[1])
        Physical_topology = copy.deepcopy(pt_res[0])

        requests = Generate(seed_no=sim_input['seed'],
                            nodes=list(sim_input['physical_topology']['nodes'].keys()),
                            holding_time_mean=sim_input['holding_time_mean'],
                            inter_arrival_time_mean=sim_input['inter_arrival_time'],
                            req_no=sim_input['number_of_request'],
                            slot_list=sim_input['BW_type'])

        sorted_request = dict(sorted(requests.items()))
        for time in sorted_request:
            if sorted_request[time]['request_type'] == "Arrival":
                rsa_res = controller_main(request_type="Release",
                                          Physical_topology=Physical_topology,
                                          network_spectrum_DB=network_spectrum_DB,
                                          slot_NO=None,
                                          path=None
                                          )
                if rsa_res is False:
                    blocking_iter += 1
                else:
                    # TODO: A starting NO reserved slot will not exist for the "arrival" case above
                    requests_status.update({sorted_request[time]['id']: {
                        "slots": rsa_res[0]['starting_NO_reserved_slot'],
                        "path": rsa_res[0]['path']
                    }})
                    network_spectrum_DB = rsa_res[1]
                    Physical_topology = rsa_res[2]
            elif sorted_request[time]['request_type'] == "Release":
                if sorted_request[time]['id'] in requests_status:
                    controller_main(request_type="Release",
                                    Physical_topology=Physical_topology,
                                    network_spectrum_DB=network_spectrum_DB,
                                    slot_NO=sorted_request[requests_status[time]['id']]['slots'],
                                    path=sorted_request[requests_status[time]['id']]['path']
                                    )

        blocking.update({i: blocking_iter / sim_input['number_of_request']})


if __name__ == '__main__':
    main()
