import networkx as nx
import numpy as np
import copy
from Request_generator import Generate
from load_input import load_input
from SDN_Controller import controller_main


blocking = {}
input = load_input()

def creat_PT(input):
    G = nx.Graph()
    spectrum_DB = {}
    for node in input['physical_topology']['nodes']:
        G.add_node(node)
    for link_no in input['physical_topology']['links']:
        spectrum_DB.update({(input['physical_topology']['links'][link_no]['source'], input['physical_topology']['links'][link_no]['destination']):np.zeros(input['number_of_slot_per_lisnk'])})
        G.add_edge(input['physical_topology']['links'][link_no]['source'], input['physical_topology']['links'][link_no]['destination'], length = input['physical_topology']['links'][link_no]['length'])
    return G, spectrum_DB



for i in range(input['NO_iteration']):
    blocking_iter = 0
    requests_status={}
    pt_res = creat_PT(input,)
    network_spectrum_DB = copy.deepcopy(pt_res[1])
    Physical_topology = copy.deepcopy(pt_res[0])
    
    requests = Generate(seed_no = input['seed'], 
                        nodes = list(input['physical_topology']['nodes'].keys()), 
                        holding_time_mean = input['holding_time_mean'], 
                        inter_arrival_time_mean = input['inter_arrival_time'], 
                        req_no = input['number_of_request'], 
                        slot_list = input['BW_type'])
    
    sorted_request = dict(sorted(requests.items()))
    for time in sorted_request:
        if sorted_request[time]['request_type'] == "Arrival":
            rsa_res = controller_main(  request_type = "Release",
                                        Physical_topology = Physical_topology,
                                        network_spectrum_DB = network_spectrum_DB,
                                        slot_NO = None,
                                        path = None
                                    )
            if rsa_res == False:
                blocking_iter += 1
            else:
                #for item in rsa_res[0]:
                requests_status.update({sorted_request[time]['id']:{
                                                                        "slots": rsa_res[0]['starting_NO_reserved_slot'],
                                                                        "path": rsa_res[0]['path']
                                                                        }})
                network_spectrum_DB = rsa_res[1]
                Physical_topology = rsa_res[2]
        elif sorted_request[time]['request_type'] == "Release":
            if sorted_request[time]['id'] in requests_status:
                controller_main(request_type = "Release",
                                Physical_topology = Physical_topology,
                                network_spectrum_DB = network_spectrum_DB,
                                slot_NO = sorted_request[requests_status[time]['id']]['slots'],
                                path = sorted_request[requests_status[time]['id']]['path']
                                )
        
    blocking.update({ i : blocking_iter/input['number_of_request']} )