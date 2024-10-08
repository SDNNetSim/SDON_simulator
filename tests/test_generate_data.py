import unittest
import json
import os
from unittest.mock import mock_open, patch

from data_scripts.generate_data import create_pt, create_bw_info


class TestGenerateData(unittest.TestCase):
    """
    Tests functions in generate_data.py script.
    """

    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps({
        "nodes": {
            "A": {"type": "CDC"},
            "B": {"type": "CDC"},
            "C": {"type": "CDC"}
        },
        "links": {
            "1": {
                "fiber": {
                    "attenuation": 4.605111673958094e-05,
                    "non_linearity": 0.0013,
                    "dispersion": 2.0393053374841523e-26,
                    "num_cores": 4,
                    "fiber_type": 0,
                    "bending_radius": 0.05,
                    "mode_coupling_co": 0.0004,
                    "propagation_const": 4000000.0,
                    "core_pitch": 4e-05
                },
                "length": 100,
                "source": "A",
                "destination": "B",
                "span_length": 100
            },
            "2": {
                "fiber": {
                    "attenuation": 4.605111673958094e-05,
                    "non_linearity": 0.0013,
                    "dispersion": 2.0393053374841523e-26,
                    "num_cores": 4,
                    "fiber_type": 0,
                    "bending_radius": 0.05,
                    "mode_coupling_co": 0.0004,
                    "propagation_const": 4000000.0,
                    "core_pitch": 4e-05
                },
                "length": 150,
                "source": "B",
                "destination": "C",
                "span_length": 100
            }
        }
    }))
    def test_create_pt(self, mock_file):
        """
        Tests creating physical topology.
        """
        cores_per_link = 4
        net_spec_dict = {('A', 'B'): 100, ('B', 'C'): 150}

        topology_dict = create_pt(cores_per_link, net_spec_dict)

        self.assertIn('nodes', topology_dict)
        self.assertIn('links', topology_dict)
        self.assertEqual(len(topology_dict['nodes']), 3)
        self.assertEqual(len(topology_dict['links']), 2)

        for _, props in topology_dict['nodes'].items():
            self.assertEqual(props, {'type': 'CDC'})

        for link_num, props in topology_dict['links'].items():
            self.assertIn('fiber', props)
            self.assertIn('length', props)
            self.assertIn('source', props)
            self.assertIn('destination', props)
            self.assertIn('span_length', props)

        exp_top_dict = json.load(mock_file())
        check_top_dict = {'nodes': exp_top_dict['nodes'], 'links': {}}
        for link_num, data_dict in exp_top_dict['links'].items():
            check_top_dict['links'][int(link_num)] = data_dict
        self.assertEqual(topology_dict, check_top_dict)

    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps({
            "example_mod_a": {
                "25": {
                    "QPSK": {
                        "max_length": 22160,
                        "slots_needed": 1
                    },
                    "16-QAM": {
                        "max_length": 9500,
                        "slots_needed": 1
                    },
                    "64-QAM": {
                        "max_length": 3664,
                        "slots_needed": 1
                    }
                },
                "50": {
                    "QPSK": {
                        "max_length": 11080,
                        "slots_needed": 2
                    },
                    "16-QAM": {
                        "max_length": 4750,
                        "slots_needed": 1
                    },
                    "64-QAM": {
                        "max_length": 1832,
                        "slots_needed": 1
                    }
                },
                "100": {
                    "QPSK": {
                        "max_length": 5540,
                        "slots_needed": 4
                    },
                    "16-QAM": {
                        "max_length": 2375,
                        "slots_needed": 2
                    },
                    "64-QAM": {
                        "max_length": 916,
                        "slots_needed": 2
                    }
                },
                "200": {
                    "QPSK": {
                        "max_length": 2770,
                        "slots_needed": 8
                    },
                    "16-QAM": {
                        "max_length": 1187,
                        "slots_needed": 4
                    },
                    "64-QAM": {
                        "max_length": 458,
                        "slots_needed": 3
                    }
                },
                "400": {
                    "QPSK": {
                        "max_length": 1385,
                        "slots_needed": 16
                    },
                    "16-QAM": {
                        "max_length": 594,
                        "slots_needed": 8
                    },
                    "64-QAM": {
                        "max_length": 229,
                        "slots_needed": 6
                    }
                }
            }
        }))
    def test_create_bw_info_yue(self, mock_file):
        """
        Tests creating Yue's bandwidth assumptions.
        """
        mod_assumption = 'example_mod_a'
        expected_bw_mod_dict = json.loads(mock_file().read())
        bw_mod_dict = create_bw_info(mod_assumption)
        self.assertEqual(bw_mod_dict, expected_bw_mod_dict[mod_assumption])

    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps({
        "example_mod_b": {
            "25": {
                "QPSK": {
                    "max_length": 20759,
                    "slots_needed": 1
                },
                "16-QAM": {
                    "max_length": 9295,
                    "slots_needed": 1
                },
                "64-QAM": {
                    "max_length": 3503,
                    "slots_needed": 1
                }
            },
            "50": {
                "QPSK": {
                    "max_length": 10380,
                    "slots_needed": 2
                },
                "16-QAM": {
                    "max_length": 4648,
                    "slots_needed": 1
                },
                "64-QAM": {
                    "max_length": 1752,
                    "slots_needed": 1
                }
            },
            "100": {
                "QPSK": {
                    "max_length": 5190,
                    "slots_needed": 3
                },
                "16-QAM": {
                    "max_length": 2324,
                    "slots_needed": 2
                },
                "64-QAM": {
                    "max_length": 876,
                    "slots_needed": 1
                }
            },
            "200": {
                "QPSK": {
                    "max_length": 2595,
                    "slots_needed": 5
                },
                "16-QAM": {
                    "max_length": 1162,
                    "slots_needed": 3
                },
                "64-QAM": {
                    "max_length": 438,
                    "slots_needed": 2
                }
            },
            "400": {
                "QPSK": {
                    "max_length": 1298,
                    "slots_needed": 10
                },
                "16-QAM": {
                    "max_length": 581,
                    "slots_needed": 5
                },
                "64-QAM": {
                    "max_length": 219,
                    "slots_needed": 4
                }
            }
        }
    }))
    def test_create_bw_info_arash(self, mock_file):
        """
        Tests creating Arash's bandwidth assumptions.
        """
        mod_assumption = 'example_mod_b'
        expected_bw_mod_dict = json.loads(mock_file().read())
        bw_mod_dict = create_bw_info(mod_assumption)
        self.assertEqual(bw_mod_dict, expected_bw_mod_dict[mod_assumption])

    def test_create_bw_info_invalid(self):
        """
        Tests an invalid bandwidth assumption.
        """
        mod_assumption = 'invalid'
        mod_assumption_path = os.path.join('tests', 'fixtures', 'test_mod_formats.json')
        with self.assertRaises(NotImplementedError):
            create_bw_info(mod_assumption, mod_assumption_path)


if __name__ == '__main__':
    unittest.main()
