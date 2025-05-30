import numpy as np
from unyt import ucross, unyt_array

from flow360.component.simulation.user_code import Expression, Variable

# ***** General principle *****
# 1. Defer evaluation of the real cross operation only to when needed (translator). This helps preserving the original user input


# pow
def cross(foo, bar):
    if isinstance(foo, np.ndarray) and isinstance(bar, np.ndarray):
        return np.cross(foo, bar)

    if isinstance(foo, np.ndarray) and isinstance(bar, unyt_array):
        return np.cross(foo, bar) * bar.units

    if isinstance(foo, np.ndarray) and isinstance(bar, unyt_array):
        return ucross(foo, bar)

    # What else than `SolverVariable`? `UserVariable`? `Expression`?
    # How to support symbolic expression now that we get rid of numpy interop?
    # Do we only support 1 layer of module?
    # Consistent serialize and deserialize
    if isinstance(foo, Variable):
        foo_length = len(foo.value)
        foo = Expression(expression=str(foo))
    else:
        foo_length = len(foo)

    if isinstance(bar, Variable):
        bar_length = len(bar.value)
        bar = Expression(expression=str(bar))
    else:
        bar_length = len(bar)

    assert foo_length == bar_length, f"Different len {foo_length} vs {bar_length}"

    if len(foo) == 3:
        return [
            bar[2] * foo[1] - bar[1] * foo[2],
            bar[0] * foo[2] - bar[2] * foo[0],
            bar[0] * foo[1] - bar[1] * foo[0],
        ]
    raise NotImplementedError()

    # foo_processed = _preprocess(foo)
    # bar_processed = _preprocess(bar)
    #
    # if len(foo_processed) == 2:
    #     return Expression(
    #         expression=bar_processed[1] * foo_processed[0] - bar_processed[0] * foo_processed[1]
    #     )
    # elif len(foo_processed) == 3:
    #     return Expression(
    #         expression=[
    #             bar_processed[2] * foo_processed[1] - bar_processed[1] * foo_processed[2],
    #             bar_processed[0] * foo_processed[2] - bar_processed[2] * foo_processed[0],
    #             bar_processed[0] * foo_processed[1] - bar_processed[1] * foo_processed[0],
    #         ]
    #     )
    # return np.cross(foo_processed, bar_processed)
