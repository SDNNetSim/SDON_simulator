import numpy as np

from sim_scripts.routing import Routing
from sim_scripts.spectrum_assignment import SpectrumAssignment


class SDNController:
    def __int__(self, req_id, network_db, topology, num_cores, path, sim_assume, src, dest, mod_formats, chosen_bw,
                max_lps):
        # TODO: Ensure all variables are updated properly, especially in the constructor
        # TODO: I'm not sure why physical topology would ever change, why return it?
        # TODO: Move variables not shared to methods and not in the constructor
        self.req_id = req_id
        self.network_db = network_db
        self.topology = topology
        self.num_cores = num_cores
        self.path = path
        self.sim_assume = sim_assume

        self.src = src
        self.dest = dest
        self.src_dest = None
        self.dest_src = None

        if self.sim_assume == 'arash':
            self.guard_band = 0
        elif self.sim_assume == 'yue':
            self.guard_band = 1
        else:
            raise NotImplementedError

        self.spectrum_obj = SpectrumAssignment()
        self.mod_formats = mod_formats
        self.chosen_bw = chosen_bw
        self.max_lps = max_lps

    def handle_release(self):
        for i in range(len(self.path) - 1):
            for core_num in range(len(self.num_cores)):
                core_arr = self.network_db[self.src_dest]['cores_matrix'][core_num]
                req_indexes = np.where(core_arr == self.req_id)
                guard_bands = np.where(core_arr == (self.req_id * -1))

                for index in req_indexes:
                    self.network_db[self.src_dest]['cores_matrix'][core_num][index] = 0
                    self.network_db[self.dest_src]['cores_matrix'][core_num][index] = 0
                for gb_index in guard_bands:
                    self.network_db[self.src_dest]['cores_matrix'][core_num][gb_index] = 0
                    self.network_db[self.dest_src]['cores_matrix'][core_num][gb_index] = 0

    def handle_arrival(self, start_slot, num_slots, core_num):
        for i in range(len(self.path) - 1):
            src_dest = (self.path[i], self.path[i + 1])
            dest_src = (self.path[i + 1], self.path[i])

            end_index = start_slot + num_slots
            # Remember, Python list indexing is up to and NOT including!
            self.network_db[src_dest]['cores_matrix'][core_num][start_slot:end_index] = self.req_id
            self.network_db[dest_src]['cores_matrix'][core_num][start_slot:end_index] = self.req_id

            # A guard band for us is a -1, as it's important to differentiate the rest of the request from it
            if self.guard_band:
                self.network_db[src_dest]['cores_matrix'][core_num][end_index] = (self.req_id * -1)
                self.network_db[dest_src]['cores_matrix'][core_num][end_index] = (self.req_id * -1)

    def allocate_lps(self):
        # TODO: Shall we only stick with one bandwidth?
        # TODO: Are we allowed to use 25 and 200?
        # No slicing is possible
        if self.chosen_bw == '25' or self.max_lps == 1:
            return False

        # Obtain the length of the path
        path_len = 0
        for i in range(len(self.path) - 1):
            path_len += self.topology[self.path[i]][self.path[i + 1]]['length']

        # TODO: Move to useful functions
        # Sort the dictionary in descending order by bandwidth
        keys_lst = [int(key) for key in self.mod_formats.keys()]
        keys_lst.sort(reverse=True)
        mod_formats = {str(i): self.mod_formats[str(i)] for i in keys_lst}

        for curr_bw, obj in mod_formats.items():
            # Cannot slice to a larger bandwidth, or slice within a bandwidth itself
            if int(curr_bw) >= int(self.chosen_bw):
                continue

            # TODO: Move to useful functions
            # Attempt to assign a modulation format
            if obj['QPSK']['max_length'] >= path_len > obj['16-QAM']['max_length']:
                tmp_format = 'QPSK'
            elif obj['16-QAM']['max_length'] >= path_len > obj['64-QAM']['max_length']:
                tmp_format = '16-QAM'
            elif obj['64-QAM']['max_length'] >= path_len:
                tmp_format = '64-QAM'
            else:
                continue

            num_slices = int(int(self.chosen_bw) / int(curr_bw))
            if num_slices > self.max_lps:
                break

            is_allocated = True
            # Check if all slices can be allocated
            for i in range(num_slices):
                self.spectrum_obj.network_spec_db = self.network_db
                self.spectrum_obj.slots_needed = obj[tmp_format]['slots_needed']
                selected_sp = self.spectrum_obj.find_free_spectrum()

                if selected_sp is not False:
                    self.handle_arrival(start_slot=selected_sp['start_slot'], num_slots=obj[tmp_format]['slots_needed'],
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
            resp = {}
            return resp, self.network_db, self.topology
        else:
            return False

    def controller_main(self, request_type):
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
            if path_mod is not False:
                slots_needed = self.mod_formats[path_mod]['slots_needed']
                self.spectrum_obj.path = selected_path
                self.spectrum_obj.slots_needed = slots_needed
                self.spectrum_obj.network_spec_db = self.network_db
                self.spectrum_obj.guard_band = self.guard_band

                selected_sp = self.spectrum_obj.find_free_spectrum()

                if selected_sp is not False:
                    resp = {}

                    self.handle_arrival(resp['start_slot'], resp['end_slot'], resp['core_num'])
                    return resp, self.network_db, self.topology
                else:
                    return self.handle_lps()
            else:
                return self.handle_lps()

        return False
