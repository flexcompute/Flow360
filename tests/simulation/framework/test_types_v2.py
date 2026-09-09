from typing import Optional

import pydantic as pd
import pytest

from flow360.component.types import Axis, Coordinate, Vector


class Model(pd.BaseModel):
    a: Optional[Axis] = pd.Field(None)
    v: Optional[Vector] = pd.Field(None)
    c: Optional[Coordinate] = pd.Field(None)


def test_axis_correct():
    a = Model(a=Axis((0, 0, 1)))
    assert type(a.a) == Axis


def test_axis_correct2():
    a = Model(a=(0, 0, 1))
    assert type(a.a) == Axis


def test_axis_incorrect():
    with pytest.raises(pd.ValidationError):
        Model(a=(0, 0, 0))


def test_axis_incorrect2():
    with pytest.raises(pd.ValidationError):
        Model(a=(0, 0, 0, 1))


def test_axis_incorrect3():
    with pytest.raises(pd.ValidationError):
        Model(a=Axis((0, 0, 0, 1)))


def test_vector_correct():
    a = Model(v=(0, 0, 1))
    assert type(a.v) == Vector


def test_vector_incorrect():
    with pytest.raises(pd.ValidationError):
        Model(v=(0, 0, 0))


def test_vector_incorrect2():
    with pytest.raises(pd.ValidationError):
        Model(v=(1, 0, 0, 0))


def test_coordinate_correct():
    a = Model(c=(0, 0, 0))
    assert type(a.c) == tuple


def test_coordinate_incorrect():
    with pytest.raises(pd.ValidationError):
        Model(c=(1, 0, 0, 0))
