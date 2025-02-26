class GNB:

    def __init__(self) -> None:
        self.link_dir = 1 # 0: downlink, 1: uplink
        self.mode = 1 # 0: TDD, 1:FDD
        self.J_carriers = 1
        self.vj_layers = 4
        self.bandwidth = 50 # MHz 
        self.sub_carrier_spacing = 30 # kHz [15, 30, 60,120]
        self.scaling_factor = 1 # [ 1, 0.8, 0.75, 0.4 ... ]
        self.frequency_range = 1 # FR1: 1, FR2: 2
        self.overhead = 0.14
        self.table_init()
        self.update_overhead()
    
    def estimate_throughputh_sinr(self, sinr):
        cqi = self.calculate_sinr2cqi(sinr)
        thr = self.estimate_throughput_cqi(cqi)
        return thr

    def estimate_throughput_cqi(self, cqi):
        if cqi == 0:
            return 0.1
        q_j = self.cqi_to_mcs[cqi]["modulation_order"]
        Rmax = self.cqi_to_mcs[cqi]["target_code_rate"] / 1024
        nbwPRBj = 0
        try:
            nbwPRBj = self.nbw_prb[self.frequency_range][self.sub_carrier_spacing][self.bandwidth]
        except:
            return 0
        T_s_u = (1e-3) / (14 * pow(2, self.frequency_range))
        thr = 0
        for j in range(self.J_carriers):
            thr += self.vj_layers*q_j*self.scaling_factor* Rmax* ((nbwPRBj*12)/T_s_u) *(1- self.overhead)
        return thr


    def set_FR_range(self, range):
        self.frequency_range = range
        self.update_overhead()
    
    def set_link_dir(self,dir):
        self.link_dir = dir
        self.update_overhead()

    def update_overhead(self):
        if self.frequency_range == 1 and self.link_dir==0:
            self.overhead=0.14
        elif self.frequency_range== 2 and self.link_dir==0:
            self.overhead = 0.18
        elif self.frequency_range ==1 and self.link_dir ==1:
            self.overhead =0.08
        elif self.frequency_range==2 and self.link_dir==1:
            self.overhead=0.1


    def calculate_sinr2cqi(self,sinr):
        """
        Calculate CQI from SINR based on typical mappings in 5G NR.

        Args:
        sinr (float): Signal-to-Interference-plus-Noise Ratio in dB.

        Returns:
        int: CQI value.
        """
        # SINR to CQI mapping table (example values)
        sinr_to_cqi = [
            (-6.7, 0), (-4.7, 1), (-2.3, 2), (0.2, 3),
            (2.4, 4), (4.3, 5), (5.9, 6), (8.1, 7),
            (10.3, 8), (11.7, 9), (14.1, 10), (16.3, 11),
            (18.7, 12), (21.0, 13), (22.7, 14), (24.5, 15)
        ]

        # Default CQI if SINR is below the minimum threshold
        cqi = 0

        # Find the appropriate CQI for the given SINR
        for threshold, cqi_value in sinr_to_cqi:
            if sinr >= threshold:
                cqi = cqi_value
            else:
                break

        return cqi




    def table_init(self):
        self.cqi_to_mcs = {
            0: {"mcs_index": None, "modulation_order": None, "target_code_rate": None},
            1: {"mcs_index": 0, "modulation_order": 2, "target_code_rate": 78},
            2: {"mcs_index": 1, "modulation_order": 2, "target_code_rate": 120},
            3: {"mcs_index": 3, "modulation_order": 2, "target_code_rate": 193},
            4: {"mcs_index": 5, "modulation_order": 2, "target_code_rate": 308},
            5: {"mcs_index": 7, "modulation_order": 2, "target_code_rate": 449},
            6: {"mcs_index": 9, "modulation_order": 2, "target_code_rate": 602},
            7: {"mcs_index": 11, "modulation_order": 4, "target_code_rate": 378},
            8: {"mcs_index": 13, "modulation_order": 4, "target_code_rate": 490},
            9: {"mcs_index": 15, "modulation_order": 4, "target_code_rate": 616},
            10: {"mcs_index": 18, "modulation_order": 6, "target_code_rate": 466},
            11: {"mcs_index": 20, "modulation_order": 6, "target_code_rate": 567},
            12: {"mcs_index": 22, "modulation_order": 6, "target_code_rate": 666},
            13: {"mcs_index": 24, "modulation_order": 6, "target_code_rate": 772},
            14: {"mcs_index": 26, "modulation_order": 6, "target_code_rate": 873},
            15: {"mcs_index": 28, "modulation_order": 6, "target_code_rate": 948},
        }

        self.nbw_prb = {
            1: {
                15: {  # Subcarrier spacing = 15 kHz
                    5: 25,   # Bandwidth = 5 MHz
                    10: 52,  # Bandwidth = 10 MHz
                    15: 79,  # Bandwidth = 15 MHz
                    20: 106, # Bandwidth = 20 MHz
                    25: 133, # Bandwidth = 25 MHz
                    30: 160, # Bandwidth = 30 MHz
                    40: 216, # Bandwidth = 40 MHz
                    50: 270  # Bandwidth = 50 MHz
                },
                30: {  # Subcarrier spacing = 30 kHz
                    10: 24,  # Bandwidth = 10 MHz
                    15: 38,  # Bandwidth = 15 MHz
                    20: 51,  # Bandwidth = 20 MHz
                    25: 65,  # Bandwidth = 25 MHz
                    30: 78,  # Bandwidth = 30 MHz
                    40: 106, # Bandwidth = 40 MHz
                    50: 133, # Bandwidth = 50 MHz
                    60: 162, # Bandwidth = 60 MHz
                    70: 189, # Bandwidth = 70 MHz
                    80: 217, # Bandwidth = 80 MHz
                    90: 245, # Bandwidth = 90 MHz
                    100: 273 # Bandwidth = 100 MHz
                },
                60: {  # Subcarrier spacing = 60 kHz
                    50: 66,  # Bandwidth = 50 MHz
                    60: 79,  # Bandwidth = 60 MHz
                    70: 93,  # Bandwidth = 70 MHz
                    80: 106, # Bandwidth = 80 MHz
                    90: 119, # Bandwidth = 90 MHz
                    100: 132 # Bandwidth = 100 MHz
                }
            },
            2: {
                60: {  # Subcarrier spacing = 60 kHz
                    50: 264, # Bandwidth = 50 MHz
                    100: 132, # Bandwidth = 100 MHz
                    200: 264, # Bandwidth = 200 MHz
                    400: 528  # Bandwidth = 400 MHz
                },
                120: {  # Subcarrier spacing = 120 kHz
                    50: 66,  # Bandwidth = 50 MHz
                    100: 132, # Bandwidth = 100 MHz
                    200: 264, # Bandwidth = 200 MHz
                    400: 528  # Bandwidth = 400 MHz
                }
            }
        }