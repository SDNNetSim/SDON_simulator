from helper_scripts.routing_helpers import get_simulated_link


def find_worst_nli(net_spec_db: dict):
    """
    Finds the maximum possible NLI for a link in the network.

    :return: The maximum NLI possible for a single link
    :rtype: float
    """
    links = list(net_spec_db.keys())
    slots_per_core = len(self.net_spec_db[links[0]]['cores_matrix'][0])
    simulated_link = get_simulated_link(slots_per_core=slots_per_core)

    original_link = copy.copy(self.net_spec_db[links[0]]['cores_matrix'][0])
    self.net_spec_db[links[0]]['cores_matrix'][0] = simulated_link

    free_channels = self.find_free_channels(net_spec_db=self.net_spec_db, slots_needed=self.slots_needed,
                                            des_link=links[0])
    taken_channels = self.find_taken_channels(net_spec_db=self.net_spec_db, des_link=links[0])

    max_spans = max(data['length'] for u, v, data in self.topology.edges(data=True)) / self.span_len
    nli_worst = self._find_link_cost(channels_dict=free_channels, taken_channels=taken_channels,
                                     num_spans=max_spans)

    self.net_spec_db[links[0]]['cores_matrix'][0] = original_link
    return nli_worst


# TODO: Potential repeat code, also move to snr_helpers
def _find_channel_mci(self, num_spans: float, center_freq: float, taken_channels: list):
    """
    For a given super-channel, calculate the multichannel interference.

    :param num_spans: The number of spans for the link.
    :type num_spans: float

    :param center_freq: The calculated center frequency of the channel.
    :type center_freq: float

    :param taken_channels: A matrix of indexes of the occupied super-channels.
    :type taken_channels: list

    :return: The calculated MCI for the channel.
    :rtype: float
    """
    mci = 0
    for channel in taken_channels:
        # The current center frequency for the occupied channel
        curr_freq = (channel[0] * self.freq_spacing) + ((len(channel) * self.freq_spacing) / 2)
        bandwidth = len(channel) * self.freq_spacing
        # Power spectral density
        power_spec_dens = self.input_power / bandwidth

        mci += (power_spec_dens ** 2) * math.log(abs((abs(center_freq - curr_freq) + (bandwidth / 2)) / (
                abs(center_freq - curr_freq) - (bandwidth / 2))))

    mci = (mci / self.mci_w) * num_spans
    return mci


# TODO: Move to snr helpers
def _find_link_cost(self, num_spans: float, channels_dict: dict, taken_channels: list):
    """
    Given a link, find the non-linear impairment cost for all cores on that link.

    :param num_spans: The number of spans for the link.
    :type num_spans: int

    :param channels_dict: The core numbers and free channels associated with it.
    :type channels_dict: dict

    :param taken_channels: The taken channels on that same link and core.
    :type taken_channels: list

    :return: The final NLI link cost calculated for the link.
    :rtype: float
    """
    # Non-linear impairment cost calculation
    nli_cost = 0

    # The total free channels found on the link
    num_channels = 0
    for core_num, free_channels in channels_dict.items():
        # Update MCI for available channel
        for channel in free_channels:
            num_channels += 1
            # Calculate the center frequency for the open channel
            center_freq = (channel[0] * self.freq_spacing) + ((self.slots_needed * self.freq_spacing) / 2)
            nli_cost += self._find_channel_mci(num_spans=num_spans, taken_channels=taken_channels[core_num],
                                               center_freq=center_freq)

    # A constant score of 1000 if the link is fully congested
    if num_channels == 0:
        return 1000.0

    link_cost = nli_cost / num_channels

    return link_cost


# TODO: Move core number to the constructor?
# TODO: Move to snr helpers
@staticmethod
def _find_adjacent_cores(core_num: int):
    """
    Given a core number, find its adjacent cores.

    :param core_num: The selected core number.
    :type core_num: int

    :param path: The path to find the NLI cost for.
    :type path: list

    :return: The final NLI cost
    :return: The indexes of the core directly before and after the selected core.
    :rtype: tuple
    """
    # Identify the adjacent cores to the currently selected core
    # The neighboring core directly before the currently selected core
    before = 5 if core_num == 0 else core_num - 1
    # The neighboring core directly after the currently selected core
    after = 0 if core_num == 5 else core_num + 1

    return before, after


# TODO: Only works for seven cores
def find_num_overlapped(channel: int, core_num: int, link_num: int):
    """
    Finds the number of overlapped channels for a single core on a link.

    :param channel: The current channel index.
    :type channel: int

    :param core_num: The current core number in the link fiber.
    :type core_num: int

    :param link_num: The current link.
    :type link_num: int

    :return: The total number of overlapped channels normalized by the number of cores.
    :rtype: float
    """
    # The number of overlapped channels
    num_overlapped = 0.0
    if core_num != 6:
        adjacent_cores = self._find_adjacent_cores(core_num=core_num)

        if self.net_spec_db[link_num]['cores_matrix'][adjacent_cores[0]][channel] > 0:
            num_overlapped += 1
        if self.net_spec_db[link_num]['cores_matrix'][adjacent_cores[1]][channel] > 0:
            num_overlapped += 1
        if self.net_spec_db[link_num]['cores_matrix'][6][channel] > 0:
            num_overlapped += 1

        num_overlapped /= 3
    # The number of overlapped cores for core six will be different (it's the center core)
    else:
        for sub_core_num in range(6):
            if self.net_spec_db[link_num]['cores_matrix'][sub_core_num][channel] > 0:
                num_overlapped += 1

        num_overlapped /= 6

    return num_overlapped


# TODO: Only works for seven cores
def find_xt_link_cost(free_slots: dict, link_num: int):
    """
    Finds the cross-talk cost for a single link.

    :param free_slots: A matrix identifying the indexes of spectral slots that are free.
    :type free_slots: dict

    :param link_num: The link number to check the cross-talk on.
    :type link_num: int

    :return: The total cross-talk value for the given link.
    :rtype float
    """
    # Non-linear impairment cost calculation
    xt_cost = 0
    # Update MCI for available channel
    num_free_slots = 0
    for core_num in free_slots:
        num_free_slots += len(free_slots[core_num])
        for channel in free_slots[core_num]:
            # The number of overlapped channels
            num_overlapped = find_num_overlapped(channel=channel, core_num=core_num, link_num=link_num)
            xt_cost += num_overlapped

    # A constant score of 1000 if the link is fully congested
    if num_free_slots == 0:
        return 1000.0

    link_cost = xt_cost / num_free_slots
    return link_cost


def nli_path(self, path: list):
    """
    Find the non-linear cost for a specific path.

    :param path: The path to find the NLI cost for.
    :type path: list

    :return: The final NLI cost
    :rtype: float
    """
    final_cost = 0
    for source, destination in zip(path, path[1:]):
        num_spans = self.topology[source][destination]['length'] / self.span_len
        link = (source, destination)

        final_cost += self._get_final_nli_cost(link=link, num_spans=num_spans, source=source,
                                               destination=destination)

    return final_cost


def get_nli_cost(link: tuple, num_spans: float, source: str, destination: str):
    """
    Controls sub-methods and calculates the final non-linear impairment cost for a link.

    :param link: The link used for calculations.
    :type link: tuple

    :param num_spans: The number of spans based on the link length.
    :type num_spans: float

    :param source: The source node.
    :type source: str

    :param destination: The destination node.
    :type destination: str

    :return: The calculated NLI cost for the link.
    :rtype: float
    """
    free_channels = find_free_channels(net_spec_db=self.net_spec_db, slots_needed=self.slots_needed,
                                       des_link=link)
    taken_channels = self.find_taken_channels(net_spec_db=self.net_spec_db, des_link=link)

    nli_cost = self._find_link_cost(num_spans=num_spans, channels_dict=free_channels,
                                    taken_channels=taken_channels)
    # Tradeoff between link length and the non-linear impairment cost
    final_cost = (self.beta * (self.topology[source][destination]['length'] / self.max_link)) + \
                 ((1 - self.beta) * nli_cost)

    return final_cost
