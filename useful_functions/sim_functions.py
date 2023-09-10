# Standard library imports
from typing import List

# Third-party library imports
import networkx as nx
import numpy as np

# Local application imports
import sim_scripts.spectrum_assignment
from sim_scripts.routing import Routing


def get_path_mod(mod_formats: dict, path_len: int):
    """
    Given an object of modulation formats and maximum lengths, choose the one that satisfies the requirements.

    :param mod_formats: The modulation object, holds needed information for maximum reach
    :type mod_formats: dict

    :param path_len: The length of the path to be taken
    :type path_len: int

    :return: The chosen modulation format, or false
    """
    if mod_formats['QPSK']['max_length'] >= path_len > mod_formats['16-QAM']['max_length']:
        resp = 'QPSK'
    elif mod_formats['16-QAM']['max_length'] >= path_len > mod_formats['64-QAM']['max_length']:
        resp = '16-QAM'
    elif mod_formats['64-QAM']['max_length'] >= path_len:
        resp = '64-QAM'
    else:
        return False

    return resp


def sort_dict_keys(dictionary: dict):
    """
    Given a dictionary with key-value pairs, return a new dictionary with the same pairs, sorted by keys in descending
    order.

    :param dictionary: The dictionary to sort.
    :type dictionary: dict

    :return: A new dictionary with the same pairs as the input dictionary, but sorted by keys in descending order.
    :rtype: dict
    """
    sorted_keys = sorted(map(int, dictionary.keys()), reverse=True)
    sorted_dict = {str(key): dictionary[str(key)] for key in sorted_keys}

    return sorted_dict


def find_path_len(path: List[str], topology: nx.Graph):
    """
    Finds the length of a path in a physical topology.

    :param path: A list of integers representing the nodes in the path.
    :type path: list of str

    :param topology: A networkx graph object representing the physical topology of the simulation.
    :type topology: networkx.Graph

    :return: The length of the path.
    """
    path_len = 0
    for i in range(len(path) - 1):
        path_len += topology[path[i]][path[i + 1]]['length']

    return path_len


# TODO: Make a plan to make these functions more efficient and integrate
# TODO: Static method
# TODO: Potentially move to useful functions
# TODO: Split into 1-3 sub methods
# TODO: May benefit from inline comments
# TODO: Variable naming
# TODO: Method name misleading
# TODO: Potentially two different methods
def find_overlapped_channel(channel_intersection, free_slots):
    # TODO: Counting overlapping in each link and slots
    overlapped_channels = {}
    non_overlapped_channels = {}
    for core_num in channel_intersection:
        overlapped_channels.update({core_num: []})
        non_overlapped_channels.update({core_num: []})
        # TODO: Enumerate and index isn't used?
        # TODO: Nested for loops most likely not needed
        # TODO: Goal is to find slot indexes (super channel) for non-overlapped channels
        for _, slot_list in enumerate(channel_intersection[core_num]):
            overlapped_cnt = 0
            for link_num in free_slots:
                for slot_num in slot_list:
                    for sub_core_num in free_slots[link_num]:
                        if core_num != sub_core_num:
                            # TODO: Comments for core_num discretion
                            if core_num == 6 and slot_num not in free_slots[link_num][sub_core_num]:
                                overlapped_cnt += 1
                            if core_num != 6:
                                before = 5 if core_num == 0 else core_num - 1
                                after = 0 if core_num == 5 else core_num + 1
                                if slot_num not in free_slots[link_num][before]:
                                    overlapped_cnt += 1
                                if slot_num not in free_slots[link_num][after]:
                                    overlapped_cnt += 1
                                if slot_num not in free_slots[link_num][6]:
                                    overlapped_cnt += 1
                    if overlapped_cnt != 0:
                        overlapped_channels[core_num].append(slot_list)
                        break
                if overlapped_cnt != 0:
                    break
            if overlapped_cnt == 0:
                non_overlapped_channels[core_num].append(slot_list)
    return non_overlapped_channels, overlapped_channels


# TODO: Function not used in spectrum assignment
# TODO: Repeat code (integrate or remove)
def _find_taken_channels(self):
    taken_channels = {}
    for source_dest in zip(self.path, self.path[1:]):
        # TODO: Check to make sure source_dest is a tuple
        taken_channels.update({source_dest: {}})
        for core_num, link in enumerate(self.net_spec_db[source_dest]['cores_matrix']):
            channels = []
            curr_channel = []

            for value in link:
                if value > 0:
                    curr_channel.append(value)
                elif value < 0 and curr_channel:
                    channels.append(curr_channel)
                    curr_channel = []

            if curr_channel:
                channels.append(curr_channel)
            taken_channels[source_dest].update({core_num: channels})

    return taken_channels


# TODO: Update docstring
# TODO: Function not used
def dict_intersection(input_dict):
    # Convert dictionary values to sets
    sets = [set(values) for values in input_dict.values()]

    # Find the intersection of the sets
    intersection_set = set.intersection(*sets)

    return intersection_set


# TODO: Similar function in engine.py, combine methods here
def find_free_slots(net_spec_db, link_num):
    link = net_spec_db[link_num]['cores_matrix']
    resp = {}
    for core_num in range(len(link)):
        indexes = np.where(link[core_num] == 0)[0]
        resp.update({core_num: indexes})

    return resp


# TODO: Integrate with find_taken_channels method in routing eventually
# TODO: Test this method (can combine to one method for efficiency)
def find_free_channels(slots_needed, free_slots):
    resp = {}
    for core_num in free_slots:
        channels = []
        curr_channel = []

        # TODO: Why enumerate if we don't use both variables?
        for index in free_slots[core_num]:
            for slot_num in range(0, slots_needed):
                # TODO: Would probably be much faster to check the range
                #  of indexes in one go instead of nested for loops
                if index + slot_num in free_slots[core_num]:
                    curr_channel.append(index + slot_num)
                else:
                    curr_channel = []
                    break
            if len(curr_channel) == slots_needed:
                channels.append(curr_channel)
                curr_channel = []

        # Check if the last group forms a subarray
        if len(curr_channel) == slots_needed:
            channels.append(curr_channel)
        resp.update({core_num: channels})

    return resp


def get_route(source, destination, topology, net_spec_db, mod_per_bw, chosen_bw, guard_slots, beta, route_method,
              ai_obj):
    routing_obj = Routing(source=source, destination=destination,
                          topology=topology, net_spec_db=net_spec_db,
                          mod_formats=mod_per_bw[chosen_bw], bandwidth=chosen_bw,
                          guard_slots=guard_slots)

    # TODO: Change constant QPSK modulation formats
    if route_method == 'nli_aware':
        slots_needed = mod_per_bw[chosen_bw]['QPSK']['slots_needed']
        routing_obj.slots_needed = slots_needed
        routing_obj.beta = beta
        selected_path, path_mod = routing_obj.nli_aware()
    elif route_method == 'xt_aware':
        # TODO: Add xt_type to the configuration file
        selected_path, path_mod = routing_obj.xt_aware(beta=beta, xt_type='with_length')
    elif route_method == 'least_congested':
        selected_path = routing_obj.least_congested_path()
        # TODO: Constant QPSK for now
        path_mod = 'QPSK'
    elif route_method == 'shortest_path':
        selected_path, path_mod = routing_obj.shortest_path()
    elif route_method == 'ai':
        # Used for routing related to artificial intelligence
        selected_path = ai_obj.route(source=int(source), destination=int(destination),
                                     net_spec_db=net_spec_db, chosen_bw=chosen_bw,
                                     guard_slots=guard_slots)

        # A path could not be found, assign None to path modulation
        if not selected_path:
            path_mod = None
        else:
            path_len = find_path_len(path=selected_path, topology=topology)
            path_mod = get_path_mod(mod_formats=mod_per_bw[chosen_bw], path_len=path_len)
    else:
        raise NotImplementedError(f'Routing method not recognized, got: {route_method}.')

    return selected_path, path_mod


def get_spectrum(mod_per_bw, chosen_bw, path, net_spec_db, guard_slots, alloc_method, modulation, check_snr, snr_obj,
                 path_mod, spectral_slots):
    slots_needed = mod_per_bw[chosen_bw][modulation]['slots_needed']
    spectrum_assignment = sim_scripts.spectrum_assignment.SpectrumAssignment(path=path, slots_needed=slots_needed,
                                                                             net_spec_db=net_spec_db,
                                                                             guard_slots=guard_slots,
                                                                             is_sliced=False, alloc_method=alloc_method)

    spectrum = spectrum_assignment.find_free_spectrum()

    if spectrum is not False:
        if check_snr:
            _update_snr_obj(snr_obj=snr_obj, spectrum=spectrum, path=path, path_mod=path_mod,
                            spectral_slots=spectral_slots, net_spec_db=net_spec_db)
            snr_check = handle_snr(check_snr=check_snr, snr_obj=snr_obj)

            if not snr_check:
                return False

        return spectrum

    return False


def _update_snr_obj(snr_obj, spectrum, path, path_mod, spectral_slots, net_spec_db):
    snr_obj.path = path
    snr_obj.path_mod = path_mod
    snr_obj.spectrum = spectrum
    snr_obj.assigned_slots = spectrum['end_slot'] - spectrum['start_slot'] + 1
    snr_obj.spectral_slots = spectral_slots
    snr_obj.net_spec_db = net_spec_db


def handle_snr(check_snr, snr_obj):
    if check_snr == "snr_calculation_nli":
        snr_check = snr_obj.check_snr()
    elif check_snr == "xt_calculation":
        snr_check = snr_obj.check_xt()
    elif check_snr == "snr_calculation_xt":
        snr_check = snr_obj.check_snr_xt()
    else:
        raise NotImplementedError(f'Unexpected check_snr flag got: {check_snr}')

    return snr_check
