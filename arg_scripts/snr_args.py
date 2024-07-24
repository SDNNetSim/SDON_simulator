class SNRProps:
    def __init__(self):
        self.light_frequency = 1.9341 * 10 ** 14  # Center light frequency
        self.plank = 6.62607004e-34  # Plank's constant
        self.req_bit_rate = 12.5  # Request bit rate
        self.req_snr = 8.5  # Request signal to noise ratio value
        self.nsp = 1.8  # Noise spectral density

        self.center_freq = None  # Center frequency for current request
        self.bandwidth = None  # Bandwidth for current request
        self.center_psd = None  # Center power spectral density for current request
        self.mu_param = None  # Mu parameter for calculating PSD
        self.sci_psd = None  # Self-channel interference PSD
        self.xci_psd = None  # Cross-channel interference PSD
        self.length = None  # Length of a current span
        self.num_span = None  # Number of span

        self.link_dict = None  # Dictionary of links for calculating various metrics
