"""
Utility functions for bet_translator_interface
"""

import operator
from math import exp, log

# pylint: disable=missing-class-docstring
class Array(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # pylint: disable=missing-function-docstring
    def operator(self, op, value):
        result = self.copy()
        for i, x in enumerate(result):
            result[i] = op(x, value)
        return Array(result)

    # pylint: disable=missing-function-docstring
    def array_operator(self, op, other):
        result = self.copy()
        for i, (x, value) in enumerate(zip(result, other)):
            result[i] = op(x, value)
        return Array(result)

    # pylint: disable=missing-function-docstring, no-self-argument
    def array_op(func):
        def wrapper(self, value):
            if isinstance(value, Array):
                # pylint: disable=not-callable
                return func(self, value, self.array_operator)
            # pylint: disable=not-callable
            return func(self, value, self.operator)

        return wrapper

    @array_op
    # pylint: disable=unexpected-special-method-signature
    def __add__(self, value, optype):
        return optype(operator.add, value)

    @array_op
    # pylint: disable=unexpected-special-method-signature
    def __sub__(self, value, optype):
        return optype(operator.sub, value)

    @array_op
    # pylint: disable=unexpected-special-method-signature
    def __mul__(self, value, optype):
        return optype(operator.mul, value)

    @array_op
    # pylint: disable=unexpected-special-method-signature
    def __truediv__(self, value, optype):
        return optype(operator.truediv, value)

    @array_op
    def __pow__(self, value, optype):
        return optype(operator.pow, value)


# pylint: disable=missing-function-docstring
def op_list(a, op, *args):
    for i, x in enumerate(a):
        a[i] = op(x, *args)
    return a


# pylint: disable=missing-function-docstring
def clip(a, a_min, a_max):
    a = op_list(a, min, a_max)
    a = op_list(a, max, a_min)
    return a


# pylint: disable=missing-function-docstring
def exp_list(a):
    return op_list(a, exp)


# pylint: disable=missing-function-docstring
def log_list(a):
    return op_list(a, log)


# pylint: disable=missing-function-docstring
def abs_list(a):
    return op_list(a, abs)
