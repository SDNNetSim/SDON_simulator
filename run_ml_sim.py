# TODO: This is the main script that will run the machine learning simulation.
#   - Helper functions in ml_helpers.py to set up models
#   - Any arguments for this script in ml_args.py
#   - The scikit learn library does it's thing
#   - I control the result output and saving.
def _run_iters():
    """
    Handles the main training or testing iterations.

    :return:
    """
    raise NotImplementedError


def _get_model():
    pass


def _print_info():
    pass


# TODO: I'm not sure if we need an environment object here.
# TODO: The if-else logic for train/test will be implemented in the _run function.
#   - Getting a trained model
#   - Running the trained model
#   - Else, training the model
#   - Calls _run_iters
def _run():
    """
    Controls the simulation of the machine learning model.

    :return: None
    """
    raise NotImplementedError


def _setup_ml_sim():
    """
    Gets the simulation input parameters.

    :return: The simulation input parameters.
    :rtype: dict
    """
    raise NotImplementedError


def run_ml_sim():
    """
    Controls the simulation of the machine learning model.

    :return: None
    """
    _setup_ml_sim()
    _run()


if __name__ == '__main__':
    run_ml_sim()
