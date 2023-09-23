import argparse

from config.arg_constants import PARAMETERS


def parse_args():
    """
    Parse and error check command line arguments passed.

    :return: An object containing the values for each valid parameter.
    :rtype: obj
    """
    parser = argparse.ArgumentParser(description='Software-Defined Networking Simulator.')

    for args_lst in PARAMETERS:
        argument, arg_type, arg_help = args_lst[0], args_lst[1], args_lst[2]
        parser.add_argument(f'--{argument}', type=arg_type, help=arg_help)

    args = parser.parse_args()

    return vars(args)
