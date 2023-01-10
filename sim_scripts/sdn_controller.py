import numpy as np

from sim_scripts.routing import Routing
from sim_scripts.spectrum_assignment import SpectrumAssignment
from useful_functions.sim_functions import get_path_mod, sort_dict_keys, find_path_len


class SDNController:
    def __init__(self, req_id=None, network_db=None, topology=None, num_cores=None, path=None, sim_assume=None,
                 src=None, dest=None, mod_formats=None, chosen_bw=None, max_lps=None):
        self.req_id = req_id
        self.network_db = network_db
        self.topology = topology
        self.num_cores = num_cores
        self.path = path
        self.sim_assume = sim_assume

        self.src = src
        self.dest = dest

        self.mod_formats = mod_formats
        self.chosen_bw = chosen_bw
        self.max_lps = max_lps

        if self.sim_assume == 'arash':
            self.guard_band = 0
        elif self.sim_assume == 'yue':
            self.guard_band = 1
        else:
            raise NotImplementedError

    def handle_release(self):
        for i in range(len(self.path) - 1):
            src_dest = (self.path[i], self.path[i + 1])
            dest_src = (self.path[i + 1], self.path[i])

            for core_num in range(self.num_cores):
                core_arr = self.network_db[src_dest]['cores_matrix'][core_num]
                req_indexes = np.where(core_arr == self.req_id)
                guard_bands = np.where(core_arr == (self.req_id * -1))

                for index in req_indexes:
                    self.network_db[src_dest]['cores_matrix'][core_num][index] = 0
                    self.network_db[dest_src]['cores_matrix'][core_num][index] = 0
                for gb_index in guard_bands:
                    self.network_db[src_dest]['cores_matrix'][core_num][gb_index] = 0
                    self.network_db[dest_src]['cores_matrix'][core_num][gb_index] = 0

    def handle_arrival(self, start_slot, end_slot, core_num):
        for i in range(len(self.path) - 1):
            src_dest = (self.path[i], self.path[i + 1])
            dest_src = (self.path[i + 1], self.path[i])

            # Remember, Python list indexing is up to and NOT including!
            self.network_db[src_dest]['cores_matrix'][core_num][start_slot:end_slot] = self.req_id
            self.network_db[dest_src]['cores_matrix'][core_num][start_slot:end_slot] = self.req_id

            # A guard band for us is a -1, as it's important to differentiate the rest of the request from it
            if self.guard_band:
                self.network_db[src_dest]['cores_matrix'][core_num][end_slot] = (self.req_id * -1)
                self.network_db[dest_src]['cores_matrix'][core_num][end_slot] = (self.req_id * -1)

    def allocate_lps(self):
        # TODO: Shall we only stick with one bandwidth?
        # TODO: Are we allowed to use 25 and 200?
        # No slicing is possible
        if self.chosen_bw == '25' or self.max_lps == 1:
            return False

        path_len = find_path_len(self.path, self.topology)
        # Sort the dictionary in descending order by bandwidth
        mod_formats = sort_dict_keys(self.mod_formats)

        for curr_bw, obj in mod_formats.items():
            # Cannot slice to a larger bandwidth, or slice within a bandwidth itself
            if int(curr_bw) >= int(self.chosen_bw):
                continue

            tmp_format = get_path_mod(obj, path_len)
            if tmp_format is False:
                continue

            num_slices = int(int(self.chosen_bw) / int(curr_bw))
            if num_slices > self.max_lps:
                break

            is_allocated = True
            # Check if all slices can be allocated
            for i in range(num_slices):
                spectrum_assignment = SpectrumAssignment(self.path, obj[tmp_format]['slots_needed'], self.network_db,
                                                         guard_band=self.guard_band)
                selected_sp = spectrum_assignment.find_free_spectrum()

                if selected_sp is not False:
                    # TODO: Update end slot
                    self.handle_arrival(start_slot=selected_sp['start_slot'], end_slot=obj[tmp_format]['slots_needed'],
                                        core_num=selected_sp['core_num'])
                # Clear all previously attempted allocations
                else:
                    self.handle_release()
                    is_allocated = False
                    break

            if is_allocated:
                return True

        return False

    def handle_lps(self):
        # TODO: Only do this if the request has been blocked? Or always?
        # TODO: Mod formats will change (Is resp even needed?)
        lps_resp = self.allocate_lps()
        if lps_resp is not False:
            resp = {
                'path': self.path,
                'is_sliced': True,
            }
            return resp, self.network_db, self.topology
        else:
            return False

    def handle_event(self, request_type):
        if request_type == "release":
            self.handle_release()
            return self.network_db, self.topology

        routing_obj = Routing(req_id=self.req_id, source=self.src, destination=self.dest,
                              physical_topology=self.topology, network_spec_db=self.network_db,
                              mod_formats=self.mod_formats, bw=self.chosen_bw)

        if self.sim_assume == 'yue':
            selected_path, path_mod = routing_obj.shortest_path()
        elif self.sim_assume == 'arash':
            selected_path = routing_obj.least_congested_path()
            path_mod = 'QPSK'
        else:
            raise NotImplementedError

        if selected_path is not False:
            self.path = selected_path
            if path_mod is not False:
                slots_needed = self.mod_formats[path_mod]['slots_needed']
                spectrum_assignment = SpectrumAssignment(self.path, slots_needed, self.network_db,
                                                         guard_band=self.guard_band)

                # TODO: Ensure spectrum assignment works correctly (correct path found)
                # TODO: End slot is not correct?
                selected_sp = spectrum_assignment.find_free_spectrum()

                if selected_sp is not False:
                    resp = {
                        'path': selected_path,
                        'mod_format': path_mod,
                        'core_num': selected_sp['core_num'],
                        'start_slot': selected_sp['start_slot'],
                        'end_slot': selected_sp['end_slot'],
                        'is_sliced': False
                    }

                    # TODO: We assume spectrum assignment work correctly here
                    # TODO: End slot should be passed here, it is not correct?
                    self.handle_arrival(selected_sp['start_slot'], selected_sp['end_slot'], selected_sp['core_num'])
                    return resp, self.network_db, self.topology
                else:
                    # return self.handle_lps()
                    return False
            else:
                # return self.handle_lps()
                return False

        return False
