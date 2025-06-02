import numpy as np
from unyt import ucross, unyt_array

# ***** General principle *****
# 1. Defer evaluation of the real cross operation only to when needed (translator). This helps preserving the original user input


# pow
def cross(foo, bar):
    """Customized Cross function to work with the `Expression` and Variables"""
    # TODO: Move global import here to avoid circular import.
    # Cannot find good way of breaking the circular import otherwise.
    print("input : \n", foo, "type = ", type(foo), "\n", bar, "type = ", type(bar), "\n")
    from flow360.component.simulation.user_code import Expression, Variable

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

    def _preprocess_input(baz):
        if isinstance(baz, Variable):
            if isinstance(baz.value, Expression):
                return _preprocess_input(baz.value)
            # value
            baz_length = len(baz.value)
            baz = Expression(expression=str(baz))
        elif isinstance(baz, Expression):
            vector_form = baz.as_vector()
            if not vector_form:  # I am scalar expression.
                raise ValueError(f"fl.cross() can not take in scalar expression. {baz} was given")

            baz_length = len(vector_form)
            baz = vector_form
        else:
            baz_length = len(baz)

        return baz, baz_length

    foo, foo_length = _preprocess_input(foo)
    bar, bar_length = _preprocess_input(bar)
    print("\n>>>> foo, foo_length = ", foo, foo_length)
    print(">>>> bar, bar_length = ", bar, bar_length)
    assert foo_length == bar_length, f"Different len {foo_length} vs {bar_length}"
    print(
        ">> HOW??? ",
        [
            bar[2] * foo[1] - bar[1] * foo[2],
            bar[0] * foo[2] - bar[2] * foo[0],
            bar[0] * foo[1] - bar[1] * foo[0],
        ],
    )
    if foo_length == 3:
        return Expression.model_validate(
            [
                bar[2] * foo[1] - bar[1] * foo[2],
                bar[0] * foo[2] - bar[2] * foo[0],
                bar[0] * foo[1] - bar[1] * foo[0],
            ]
        )
    raise NotImplementedError("len ==2 not implemented")

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
