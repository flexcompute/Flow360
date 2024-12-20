import operator
from math import *


class array(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def operator(self, op, value):
        result = self.copy()
        for i, x in enumerate(result):
            result[i] = op(x, value)
        return array(result)

    def arrayOperator(self, op, other):
        result = self.copy()
        for i, (x, value) in enumerate(zip(result, other)):
            result[i] = op(x, value)
        return array(result)

    def arrayOp(func):
        def wrapper(self, value):
            if isinstance(value, array):
                return func(self, value, self.arrayOperator)
            return func(self, value, self.operator)

        return wrapper

    @arrayOp
    def __add__(self, value, optype):
        return optype(operator.add, value)

    @arrayOp
    def __sub__(self, value, optype):
        return optype(operator.sub, value)

    @arrayOp
    def __mul__(self, value, optype):
        return optype(operator.mul, value)

    @arrayOp
    def __truediv__(self, value, optype):
        return optype(operator.truediv, value)

    @arrayOp
    def __pow__(self, value, optype):
        return optype(operator.pow, value)


def opList(a, op, *args):
    for i, x in enumerate(a):
        a[i] = op(x, *args)
    return a


def clip(a, a_min, a_max):
    a = opList(a, min, a_max)
    a = opList(a, max, a_min)
    return a


def expList(a):
    return opList(a, exp)


def logList(a):
    return opList(a, log)


def absList(a):
    return opList(a, abs)
