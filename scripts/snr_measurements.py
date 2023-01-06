import math

import numpy as np



class SnrMeasurments:
    """
    Calculates SNR for a given request.
    """
    
    def __init__(self, path, modulation_format, start_slot_no, end_slot_no, no_assigned_slots, 
                 assigned_core_no, requested_bit_rate, physical_topology, frequncy_spacing, input_power, 
                 spectral_slots, SNR_requested, requests_status = None, phi = None, network_spec_db=None, 
                 guard_band=0, baud_rates = None, EGN = None):
        self.path = path

        self.no_assigned_slots = no_assigned_slots
        self.start_slot_no = start_slot_no
        self.end_slot_no = end_slot_no
        self.assigned_core_no = assigned_core_no
        self.requested_bit_ratr = requested_bit_rate
        self.modulation_format = modulation_format
        self.guard_band = guard_band
        self.physical_topology = physical_topology
        self.network_spec_db = network_spec_db
        self.frequncy_spacing = frequncy_spacing
        self.input_power = input_power
        self.spectral_slots = spectral_slots
        self.SNR_requested = SNR_requested
        self.phi = phi
        self.EGN = EGN
        self.requests_status = requests_status
        self.baud_rates = baud_rates
        
        self.cores_matrix = None
        self.rev_cores_matrix = None
        self.num_slots = None

        self.response = {'SNR': None }
        
        
    def G_NLI_ASE(self):
        light_frequncy = (1.9341 * math.e ** 14)
        Rho = ( ( math.pi ** 2 ) * np.abs( self.physical_topology['dispersion'] ) )/( 2 * self.physical_topology['attenuation'])
        Mio = ( 3 * ( self.physical_topology['non_linearity'] ** 2 ) ) / ( 2 * math.pi * self.physical_topology['attenuation'] * np.abs( self.physical_topology['dispersion'] ))
        Fi = (self.start_slot_no * self.frequncy_spacing ) + ( ( self.no_assigned_slots * self.frequncy_spacing ) / 2 )
        BW = self.no_assigned_slots * self.frequncy_spacing
        PSDi = self.input_power / BW
        SCI = (PSDi ** 2) * math.asinh( Rho * (BW ** 2 ) )
        PSD_NLI = 0
        PSD_corr = 0
        for link in self.path:
            MCI = 0
            Num_span = 0
            visited_channel = []
            for w in range(self.spectral_slots):
                if self.network_spec_db[link][self.assigned_core_no][w] != 0 :
                    if self.network_spec_db[link][self.assigned_core_no][w] in visited_channel:
                        continue
                    else:
                        visited_channel.append(self.network_spec_db[link][self.assigned_core_no][w])
                        
                    Fj = ( w * self.frequncy_spacing)+((self.requests_status[self.network_spec_db[link][self.assigned_core_no][w]]['slot_number'] * self.frequncy_spacing ) / 2 )
                    BWj = self.requests_status[self.network_spec_db[link][self.assigned_core_no][w]]['slot_number'] * self.frequncy_spacing
                    PSDj = self.input_power / BWj
                    if Fi != Fj:
                        MCI = MCI + ((PSDj ** 2) * math.log( math.abs ((math.abs(Fi-Fj)+(BWj/2))/(math.abs(Fi-Fj)-(BWj/2)))))
            
            if self.phi:
                hn = math.ceil( ( len(visited_channel) - 1 ) / 2 )
                for i in range(1,):
                    hn = hn + 1 / i
                effective_L = ( 1 - math.e ** ( -2 * self.physical_topology['attenuation'] * self.physical_topology['span_length']) ) / ( 2 * self.physical_topology['attenuation'])
                baud_rate = self.baud_rates[self.modulation_format]
                temp_coef = ((self.physical_topology['non_linearity'] ** 2 ) * (effective_L ** 2) * (PSDi ** 3) *  ( BW ** 2 ) ) / ( ( baud_rate ** 2 ) * math.pi * self.physical_topology['dispersion'] * self.physical_topology['span_length'])
                PSD_corr = ( 80 / 81 ) * self.phi[self.modulation_format] * temp_coef * hn
            
            
            PSD_ASE = 0
            if self.EGN:
                PSD_NLI = ( ( ( SCI + MCI ) * Mio * PSDi) ) - PSD_corr
            else:
                PSD_NLI = ( ( ( SCI + MCI ) * Mio * PSDi) )
            PSD_ASE = ( self.physical_topology['plank'] * light_frequncy * self.physical_topology['nsp'] ) * ( math.e ** (self.physical_topology['attenuation'] * self.physical_topology['span_length'] * math.e ** 3 ) - 1 )
            # SNR =( 1 / ( PSDi / ( ( PSD_ASE + PSD_NLI ) * Num_span ) ) )
            for i in range(1,100):
                Num_span =  i
                SNR = ( 1 / ( PSDi / ( ( PSD_ASE + PSD_NLI ) * Num_span ) ) )
                SNR2 = 10*math.log(1/SNR)
                if SNR2 < self.SNR_requested:
                    print( "Maximum distance  " + (i-1) * self.physical_topology['span_length'] )
                


            
            