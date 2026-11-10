"""
utilities module
"""

import time
from functools import wraps


# pylint: disable=invalid-name
class classproperty(property):
    """classproperty decorator"""

    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


def measure_execution_time(func):
    """
    Decorator to measure the execution time of a function.

    Parameters
    ----------
    func : function
        The function whose execution time is to be measured.

    Returns
    -------
    wrapper : function
        The wrapped function with added execution time measurement.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.time()  # Record the end time
        execution_time = end_time - start_time  # Calculate the execution time
        print(f"Execution time of {func.__name__}: {execution_time:.4f} seconds")
        return result

    return wrapper
