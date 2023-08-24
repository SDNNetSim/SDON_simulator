import math

import numpy as np



class SnrMeasurments:
    """
    Calculates SNR for a given request.
    """
    
    def __init__(self, 
                 path, 
                 modulation_format, 
                 SP, 
                 no_assigned_slots, 
                 requested_bit_rate, 
                 frequncy_spacing, 
                 input_power, 
                 spectral_slots, 
                 requested_SNR, 
                 network_spec_db,
                 physical_topology,
                 requests_status = None, 
                 phi = None,
                 guard_band = 0, 
                 baud_rates = None, 
                 EGN = None,  
                 XT_noise = False, 
                 bidirectional = True):
        
        self.path = path
        self.SP = SP
        self.no_assigned_slots = no_assigned_slots
        self.requested_bit_rate = requested_bit_rate
        self.modulation_format = modulation_format
        self.guard_band = guard_band
        self.frequncy_spacing = frequncy_spacing
        self.input_power = input_power
        self.spectral_slots = spectral_slots
        self.requested_SNR = requested_SNR
        self.network_spec_db = network_spec_db
        self.physical_topology = physical_topology
        self.phi = phi
        self.EGN = EGN
        self.requests_status = requests_status
        self.baud_rates = baud_rates
        self.bidirectional = bidirectional
        self.XT_noise = XT_noise
        self.plank = 6.62607004e-34
        # self.mode_coupling_co = mode_coupling_co
        # self.bending_radius = bending_radius
        # self.propagation_const = propagation_const
        # self.core_pitch = core_pitch
        
        self.cores_matrix = None
        self.rev_cores_matrix = None
        self.num_slots = None

        self.response = {'SNR': None }
    def _find_taken_channels(self, link_num: tuple):
        """
        Finds the number of taken channels on any given link.

        :param link_num: The link number to search for channels on.
        :type link_num: int

        :return: A matrix containing the indexes to occupied or unoccupied super channels on the link.
        :rtype: list
        """
        channels = []
        curr_channel = []
        link = self.network_spec_db[link_num]['cores_matrix'][0]

        for value in link:
            if value > 0:
                curr_channel.append(value)
            elif value < 0 and curr_channel:
                channels.append(curr_channel)
                curr_channel = []

        if curr_channel:
            channels.append(curr_channel)

        return channels
    def _SCI_calculator(self, link_id, PSDi, BW):
        Rho = ( ( math.pi ** 2 ) * np.abs( self.physical_topology['links'][link_id]['fiber']['dispersion'] ) )/( 2 * self.physical_topology['links'][link_id]['fiber']['attenuation'])
        G_SCI = (PSDi ** 2) * math.asinh( Rho * (BW ** 2 ) )
        return G_SCI
    
    
    def _XCI_calculator(self, Fi, link):
        visited_channel = []
        MCI = 0
        for w in range(self.spectral_slots):
            if self.network_spec_db[(self.path[link], self.path[link+1])]['cores_matrix'][self.SP['core_num']][w] > 0: #!= 0 :
                if self.network_spec_db[(self.path[link], self.path[link+1])]['cores_matrix'][self.SP['core_num']][w] in visited_channel:
                    continue
                else:
                    visited_channel.append(self.network_spec_db[(self.path[link], self.path[link+1])]['cores_matrix'][self.SP['core_num']][w])
                BW_J = len(np.where(self.network_spec_db[(self.path[link], self.path[link+1])]['cores_matrix'][self.SP['core_num']][w] == self.network_spec_db[(self.path[link], self.path[link+1])]['cores_matrix'][self.SP['core_num']])[0]) * self.frequncy_spacing    
                Fj = (( w * self.frequncy_spacing)+((BW_J ) / 2 ) )* 10 ** 9
                BWj = BW_J * 10 **9
                PSDj = self.input_power / BWj
                if Fi != Fj:
                    MCI = MCI + ((PSDj ** 2) * math.log( abs((abs(Fi-Fj)+(BWj/2))/(abs(Fi-Fj)-(BWj/2)))))
        return MCI, visited_channel
    
    def _XT_calculator(self, link_id, length):
        XT_lambda = (2 * self.physical_topology['links'][link_id]['fiber']["bending_radius"] * self.physical_topology['links'][link_id]['fiber']["mode_coupling_co"] ** 2) / (self.physical_topology['links'][link_id]['fiber']["propagation_const"] * self.physical_topology['links'][link_id]['fiber']["core_pitch"])
        no_adjacent_core = 6
        lng = length*10*10**3
        XT_calc = (no_adjacent_core * ( 1 - math.exp(-(no_adjacent_core+1)*2*XT_lambda*lng))) / (1 + no_adjacent_core * math.exp(-(no_adjacent_core+1)*2*XT_lambda*lng))
        P_XT = no_adjacent_core * XT_lambda * length * 10**3 * self.input_power  
        return  P_XT  
    def SNR_check_NLI_ASE_XT(self):
        
        light_frequncy = (1.9341 * 10 ** 14)
        
        Fi = ((self.SP['start_slot'] * self.frequncy_spacing ) + ( ( self.no_assigned_slots * self.frequncy_spacing ) / 2 )) * 10 ** 9
        BW = self.no_assigned_slots * self.frequncy_spacing * 10 ** 9
        PSDi = self.input_power / BW
        PSD_NLI = 0
        PSD_corr = 0
        SNR = 0 
        for link in range(0, len(self.path)-1):
            MCI = 0
            Num_span = 0
            #visited_channel = []
            link_id = self.network_spec_db[(self.path[link], self.path[link+1])]['link_num']
            taken_channels = self._find_taken_channels((self.path[link], self.path[link+1]))
            if len(taken_channels) > 0:
                print("5")
            Mio = ( 3 * ( self.physical_topology['links'][link_id]['fiber']['non_linearity'] ** 2 ) ) / ( 2 * math.pi * self.physical_topology['links'][link_id]['fiber']['attenuation'] * np.abs( self.physical_topology['links'][link_id]['fiber']['dispersion'] ))
            G_SCI = self._SCI_calculator(link_id, PSDi, BW)
            G_XCI, visited_channel = self._XCI_calculator( Fi, link)
            length = self.physical_topology['links'][link_id]['span_length']
            nsp = 1.8 # TODO self.physical_topology['links'][link_id]['fiber']['nsp']
            bending_radius = 0.05 # TODO: self.physical_topology['links'][link_id]['fiber']["bending_radius"] 
            Num_span = self.physical_topology['links'][link_id]['length'] / length
            
            if self.phi:
                hn = 0
                for i in range(1,math.ceil( ( len(visited_channel) - 1 ) / 2 )+1):
                    hn = hn + 1 / i
                effective_L = ( 1 - math.e ** ( -2 * self.physical_topology['links'][link_id]['fiber']['attenuation'] * length * 10**3) ) / ( 2 * self.physical_topology['links'][link_id]['fiber']['attenuation'])
                baud_rate = int(self.requested_bit_rate) *10**9 / 2 #self.baud_rates[self.modulation_format]
                temp_coef = ((self.physical_topology['links'][link_id]['fiber']['non_linearity'] ** 2 ) * (effective_L ** 2) * (PSDi ** 3) *  ( BW ** 2 ) ) / ( ( baud_rate ** 2 ) * math.pi * self.physical_topology['links'][link_id]['fiber']['dispersion'] * (length*10**3))
                PSD_corr = ( 80 / 81 ) * self.phi[self.modulation_format] * temp_coef * hn
            
            
            PSD_ASE = 0
            if self.EGN:
                PSD_NLI = ( ( ( G_SCI + G_XCI ) * Mio * PSDi) ) - PSD_corr
            else:
                PSD_NLI = ( ( ( G_SCI + G_XCI ) * Mio * PSDi) )
            PSD_ASE = ( self.plank * light_frequncy * nsp ) * ( math.exp(self.physical_topology['links'][link_id]['fiber']['attenuation']  * length * 10 ** 3 ) - 1 )
            SNR +=( 1 / ( PSDi / ( ( PSD_ASE + PSD_NLI ) * Num_span ) ) )

            #for i in range(1,100):
            #P_XT2 = self.input_power * math.exp(-)
        SNR = 10 * math.log10( 1 / SNR ) 
        print(SNR)
        return True if SNR > self.requested_SNR else False

        """
        for i in range(1,100):
            Num_span =  i
            lng2 = Num_span * length 
            P_XT2 = no_adjacent_core * XT_lambda * self.input_power * math.exp(-self.physical_topology['links'][link_id]['fiber']['attenuation'] * lng2) * lng2 * 10**3
            P_XT2 = P_XT2 * self.no_assigned_slots
            SNR = ( 1 / ( PSDi*BW / ( ( PSD_ASE*BW + PSD_NLI*BW ) * Num_span + P_XT2 ) ) )
            SNR2 = 10*math.log10(1/SNR) 
            if self.modulation_format == '64-QAM':
                snr_tr = 22#13.5
            elif self.modulation_format == '16-QAM':
                snr_tr = 16 #9.5
            elif self.modulation_format == 'QPSK':
                snr_tr = 7.5
            if SNR2 < snr_tr:
                print( "Maximum distance:  " , (i-1) * length )
                break
        """
                


            
            