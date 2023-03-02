import numpy as np


def set_seed(seed_no):
    """
    Sets the seed for random number generation functions.

    :param seed_no: The seed
    :type seed_no: int
    :return: None
    """
    np.random.seed(seed_no)


def uniform_rv(scale_param=None):
    """
    Generates a value from a uniform distribution. Optional scale parameter.

    :param scale_param: A scale parameter
    :type scale_param: float
    :return: A uniform random variable
    :rtype: int
    """
    if scale_param is None:
        return np.random.uniform(0, 1)

    return int(np.random.uniform(0, 1) * scale_param)


def exponential_rv(scale_param):
    """
    Generates a value from an exponential distribution.

    :param scale_param: A scale parameter
    :type scale_param: float
    :return: An exponential random variable
    :rtype: float
    """
    # np.log is the natural logarithm
    return ((-1.0) / float(scale_param)) * np.log(uniform_rv())
