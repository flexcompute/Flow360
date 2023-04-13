from typing import Optional

import pytest
from pydantic import BaseModel, ValidationError

from flow360.component.types import Axis, Coordinate, Size, Vector


class Model(BaseModel):
    a: Optional[Axis]
    v: Optional[Vector]
    c: Optional[Coordinate]
    size: Optional[Size]


def test_axis_correct():
    a = Model(a=Axis((0, 0, 1)))
    assert type(a.a) == Axis


def test_axis_correct2():
    a = Model(a=(0, 0, 1))
    assert type(a.a) == Axis


def test_axis_incorrect():
    with pytest.raises(ValidationError):
        a = Model(a=(0, 0, 0))


def test_axis_incorrect2():
    with pytest.raises(ValidationError):
        a = Model(a=(0, 0, 0, 1))


def test_axis_incorrect3():
    with pytest.raises(ValidationError):
        a = Model(a=Axis((0, 0, 0, 1)))


def test_vector_correct():
    a = Model(v=(0, 0, 1))
    assert type(a.v) == Vector


def test_vector_incorrect():
    with pytest.raises(ValidationError):
        a = Model(v=(0, 0, 0))


def test_vector_incorrect2():
    with pytest.raises(ValidationError):
        a = Model(v=(1, 0, 0, 0))


def test_coordinate_correct():
    a = Model(c=(0, 0, 0))
    assert type(a.c) == tuple


def test_coordinate_incorrect():
    with pytest.raises(ValidationError):
        a = Model(c=(1, 0, 0, 0))


def test_size_correct():
    a = Model(size=(1, 1, 1))


def test_size_correct1():
    a = Model(size=[1, 1, 1])


def test_size_incorrect():
    with pytest.raises(ValidationError):
        a = Model(size=(0, 0, 0))


def test_size_incorrect1():
    with pytest.raises(ValidationError):
        a = Model(size=[0, 0, 0])


def test_size_incorrect2():
    with pytest.raises(ValidationError):
        a = Model(size=(-1, 1, 1))


def test_size_correct3():
    with pytest.raises(ValidationError):
        a = Model(size=(1, 1, 1, 1))


def test_size_correct4():
    with pytest.raises(ValidationError):
        a = Model(size=[1, 1, 1, 1])
