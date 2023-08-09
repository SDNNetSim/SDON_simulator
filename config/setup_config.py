import configparser

from config import config_constants


def _copy_dict_vals(dest_key: str, dictionary: dict):
    """
    Given a dictionary, copy the values from key t1 to a given key.

    :param dest_key: The destination key where the values will be copied to.
    :type dest_key: str

    :param dictionary: The dictionary containing t1 and its values.
    :type dictionary: dict

    :return: An updated dictionary with the values from t1 copied to a custom key in the dictionary.
    :rtype: dict
    """
    dictionary[dest_key] = dict()
    for key, val in dictionary['t1'].items():
        dictionary[dest_key][key] = val

    return dictionary


def _setup_threads(config: configparser.ConfigParser, config_dict: dict, sections: list, option_types: dict):
    """
    Checks if multiple threads should be run. If so, structure each threads' sim params.

    :param config: The configuration object
    :type config: config.ConfigParser

    :param config_dict: A dictionary containing the main thread with key value pairs.
    :type config_dict: dict

    :param sections: A list of every section in the ini file.
    :type sections: list

    :param option_types: A dictionary of all options along with their desired conversion type.
    :type option_types: dict

    :return: An updated dictionary with parameters for every thread if they exist.
    :rtype: dict
    """
    for new_thread in sections:
        config_dict = _copy_dict_vals(dest_key=new_thread, dictionary=config_dict)
        # Make desired changes for this thread
        for key, value in config.items(new_thread):
            target_type = option_types[key]
            config_dict[new_thread][key] = target_type(value)

    return config_dict


def read_config():
    """
    Reads and structures necessary data from the configuration file in the run_ini directory.
    """
    config_dict = {'t1': {}}
    config = configparser.ConfigParser()

    try:
        config.read('config/run_ini/config.ini')

        if not config.has_section('t1') or not config.has_option('t1', 'sim_type'):
            raise ValueError("Missing 't1' section in the configuration file or simulation type option.")

        if config['t1']['sim_type'] == 'arash':
            required_options = config_constants.ARASH_REQUIRED_OPTIONS
        else:
            required_options = config_constants.YUE_REQUIRED_OPTIONS

        for option in required_options:
            if not config.has_option('t1', option):
                raise ValueError(f"Missing '{option}' in the 't1' section.")

        # Structure config value into a dictionary for the main thread
        for key, value in config.items('t1'):
            # Convert the values in ini value to desired types since they default as strings
            target_type = required_options[key]
            config_dict['t1'][key] = target_type(value)

        # Init other options to None if they haven't been specified
        for other_option in config_constants.OTHER_OPTIONS:
            if other_option not in config_dict['t1']:
                config_dict['t1'][other_option] = None

        # Ignoring index zero since we've already handled t1, the first section
        resp = _setup_threads(config=config, config_dict=config_dict, sections=config.sections()[1:],
                              option_types=required_options)
        return resp

    except configparser.Error as error:
        print(f"Error reading configuration file: {error}")
        return None


if __name__ == '__main__':
    read_config()
