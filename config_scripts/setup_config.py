import configparser

from helper_scripts.os_helpers import create_dir
from arg_scripts.config_args import YUE_REQUIRED_OPTIONS, ARASH_REQUIRED_OPTIONS, OTHER_OPTIONS


def _copy_dict_vals(dest_key: str, dictionary: dict):
    """
    Given a dictionary, copy the values from key s1 to a given simulation key.

    :param dest_key: The destination key where the values will be copied to.
    :param dictionary: The original s1 dictionary.
    :return: All values of s1 copied to the given simulation key.
    :rtype: dict
    """
    dictionary[dest_key] = dict()
    for key, val in dictionary['s1'].items():
        dictionary[dest_key][key] = val

    return dictionary


def _setup_threads(config: configparser.ConfigParser, config_dict: dict, section_list: list, types_dict: dict,
                   other_dict: dict, args_obj: dict):
    """
    Checks if multiple threads/simulations should be run, structure each simulation's parameters.

    :param config: The configuration object.
    :param config_dict: The main simulations configuration params.
    :param section_list: Every section in the ini file.
    :param types_dict: Contains option conversion types.
    :param other_dict: Contains non-required options.
    :param args_obj: Arguments passed via the command line (if any).
    :return: Every simulation's structured parameters.
    :rtype: dict
    """
    for new_thread in section_list:
        config_dict = _copy_dict_vals(dest_key=new_thread, dictionary=config_dict)
        # Make desired changes for this thread
        for key, value in config.items(new_thread):
            try:
                type_obj = types_dict[key]
            except KeyError:
                type_obj = other_dict[key]
            config_dict[new_thread][key] = type_obj(value)
            # TODO: Only support for changing all s<values> as of now
            if args_obj[key] is not None:
                config_dict[new_thread][key] = args_obj[key]

    return config_dict


def read_config(args_obj: dict):
    """
    Structures necessary data from the configuration file in the run_ini directory.

    :param args_obj: Arguments passed via the command line (if any).
    :type args_obj: dict
    """
    config_dict = {'s1': dict()}
    config = configparser.ConfigParser()

    try:
        config.read('config_scripts/run_ini/config.ini')

        if not config.has_section('s1') or not config.has_option('general_settings', 'sim_type'):
            create_dir('config_scripts/run_ini')
            raise ValueError("Missing 'general_settings' section in the configuration file. "
                             "Please ensure you have a file called config.ini in the run_ini directory.")

        if config['general_settings']['sim_type'] == 'arash':
            required_dict = ARASH_REQUIRED_OPTIONS
        else:
            required_dict = YUE_REQUIRED_OPTIONS
        other_dict = OTHER_OPTIONS

        for option in required_dict:
            if not config.has_option('s1', option):
                raise ValueError(f"Missing '{option}' in the 's1' section.")

        # Structure config_scripts value into a dictionary for the main simulation
        for key, value in config.items('s1'):
            # Convert the values in ini value to desired types since they default as strings
            try:
                type_obj = required_dict[key]
                config_dict['s1'][key] = type_obj(value)
            except KeyError:
                type_obj = other_dict[key]
                config_dict['s1'][key] = type_obj(value)

            # TODO: Only support for changing all s<values> as of now
            if args_obj[key] is not None:
                config_dict['s1'][key] = args_obj[key]

        # Init other options to None if they haven't been specified
        for option in other_dict:
            if option not in config_dict['s1']:
                config_dict['s1'][option] = None

        # Ignoring index zero since we've already handled s1, the first simulation
        resp = _setup_threads(config=config, config_dict=config_dict, section_list=config.sections()[1:],
                              types_dict=required_dict, other_dict=other_dict, args_obj=args_obj)

        return resp

    except configparser.Error as error:
        print(f"Error reading configuration file: {error}")
        return None


if __name__ == '__main__':
    read_config(args_obj={'Test': None})
