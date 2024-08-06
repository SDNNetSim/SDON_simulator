SETTINGS_CONFIG_DICT = [
    {
        "category": "General",
        "settings": [
            {"type": "combo", "label": "Sim Type:", "default": "yue", "options": ["yue", "arash"]},
            {"type": "double_spin", "label": "Holding Time:", "default": 0.2, "min": 0.0, "step": 0.1},
            {"type": "spin", "label": "Arrival Rate Start:", "default": 2},
            {"type": "spin", "label": "Arrival Rate Stop:", "default": 4},
            {"type": "spin", "label": "Arrival Rate Step:", "default": 2},
            {"type": "check", "label": "Thread Erlangs:", "default": False},
            {"type": "spin", "label": "Guard Slots:", "default": 1, "min": 1},
            {"type": "spin", "label": "Number of Requests:", "default": 10000, "max": 100000},
            {"type": "line", "label": "Request Distribution:",
             "default": "{\"25\": 0.0, \"50\": 0.3, \"100\": 0.5, \"200\": 0.0, \"400\": 0.2}"},
            {"type": "spin", "label": "Max Iters:", "default": 10, "min": 1},
            {"type": "spin", "label": "Max Segments:", "default": 1, "min": 1},
            {"type": "check", "label": "Dynamic LPS:", "default": False},
            {"type": "combo", "label": "Allocation Method:", "default": "best_fit",
             "options": ["best_fit", "first_fit", "last_fit", "priority_first", "priority_last", "xt_aware"]},
            {"type": "spin", "label": "K Paths:", "default": 1, "min": 1},
            {"type": "combo", "label": "Route Method:", "default": "nli_aware",
             "options": ["nli_aware", "xt_aware", "least_congested", "shortest_path", "k_shortest_path"]},
            {"type": "check", "label": "Save Snapshots:", "default": False},
            {"type": "spin", "label": "Snapshot Step:", "default": 10, "min": 1},
            {"type": "spin", "label": "Print Step:", "default": 1, "min": 1}
        ]
    },
    {
        "category": "Topology",
        "settings": [
            {"type": "line", "label": "Network:", "default": "USNet"},
            {"type": "spin", "label": "Spectral Slots:", "default": 128},
            {"type": "double_spin", "label": "BW per Slot:", "default": 12.5},
            {"type": "spin", "label": "Cores per Link:", "default": 1},
            {"type": "check", "label": "Const Link Weight:", "default": False}
        ]
    },
    {
        "category": "SNR",
        "settings": [
            {"type": "line", "label": "SNR Type:", "default": "None"},
            {"type": "line", "label": "XT Type:", "default": "without_length"},
            {"type": "double_spin", "label": "Beta:", "default": 0.5},
            {"type": "double_spin", "label": "Theta:", "default": 0.0},
            {"type": "double_spin", "label": "Input Power:", "default": 0.001},
            {"type": "check", "label": "EGN Model:", "default": False},
            {"type": "line", "label": "Phi:",
             "default": "{\"QPSK\": 1, \"16-QAM\": 0.68, \"64-QAM\": 0.6190476190476191}"},
            {"type": "check", "label": "Bi-Directional:", "default": True},
            {"type": "check", "label": "XT Noise:", "default": False},
            {"type": "line", "label": "Requested XT:",
             "default": "{\"QPSK\": -18.5, \"16-QAM\": -25.0, \"64-QAM\": -34.0}"}
        ]
    },
    {
        "category": "File",
        "settings": [
            {"type": "line", "label": "File Type:", "default": "json"}
        ]
    }
]
