import os
import tempfile
from abc import ABCMeta
from numbers import Number

import numpy as np
import pytest
import unyt

from flow360.component.simulation.framework.entity_base import EntityBase
from flow360.component.simulation.framework.entity_registry import EntityRegistry


def _approx_equal(a, b, rel_tol=1e-12):
    """Recursively compare nested structures with float tolerance."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(_approx_equal(a[k], b[k], rel_tol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_approx_equal(ai, bi, rel_tol) for ai, bi in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, Number) and isinstance(b, Number):
        if a == b:
            return True
        return abs(a - b) <= rel_tol * max(abs(a), abs(b))
    return a == b


def to_file_from_file_test_approx(obj):
    """v2 serialization round-trip test with float tolerance."""
    test_extentions = ["yaml", "json"]
    factory = obj.__class__
    with tempfile.TemporaryDirectory() as tmpdir:
        for ext in test_extentions:
            obj_filename = os.path.join(tmpdir, f"obj.{ext}")
            obj.to_file(obj_filename)
            obj_read = factory.from_file(obj_filename)
            assert _approx_equal(obj.model_dump(), obj_read.model_dump())
            obj_read = factory(filename=obj_filename)
            assert _approx_equal(obj.model_dump(), obj_read.model_dump())


class AssetBase(metaclass=ABCMeta):
    internal_registry: EntityRegistry

    def __init__(self):
        self.internal_registry = EntityRegistry()

    def __getitem__(self, key: str) -> list[EntityBase]:
        """Use [] to access the registry"""
        if isinstance(key, str) == False:
            raise ValueError(f"Entity naming pattern: {key} is not a string.")
        found_entities = self.internal_registry.find_by_naming_pattern(key)
        if found_entities == []:
            raise ValueError(
                f"Failed to find any matching entity with {key}. Please check your input."
            )
        if len(found_entities) == 1:
            return found_entities[0]

        return found_entities


@pytest.fixture()
def array_equality_override():
    original_unyt_eq = unyt.unyt_array.__eq__
    original_unyt_ne = unyt.unyt_array.__ne__

    def unyt_array_eq(self: unyt.unyt_array, other: unyt.unyt_array):
        if isinstance(self, unyt.unyt_quantity):
            return np.ndarray.__eq__(self, other)
        if self.size == other.size:
            return all(self[i] == other[i] for i in range(len(self)))
        return False

    def unyt_array_ne(self: unyt.unyt_array, other: unyt.unyt_array):
        if isinstance(self, unyt.unyt_quantity):
            return np.ndarray.__ne__(self, other)
        if self.size == other.size:
            return any(self[i] != other[i] for i in range(len(self)))
        return True

    unyt.unyt_array.__eq__ = unyt_array_eq
    unyt.unyt_array.__ne__ = unyt_array_ne

    yield

    unyt.unyt_array.__eq__ = original_unyt_eq
    unyt.unyt_array.__ne__ = original_unyt_ne
