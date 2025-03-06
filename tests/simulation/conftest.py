from abc import ABCMeta

import numpy as np
import pytest
import unyt

from flow360.component.simulation import unit_system
from flow360.component.simulation.framework.entity_base import EntityBase
from flow360.component.simulation.framework.entity_registry import EntityRegistry


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
    # Save original methods
    original_unyt_eq = unyt.unyt_array.__eq__
    original_unyt_ne = unyt.unyt_array.__ne__
    original_flow360_eq = unit_system._Flow360BaseUnit.__eq__
    original_flow360_ne = unit_system._Flow360BaseUnit.__ne__

    # Overload equality for unyt arrays
    def unyt_array_eq(self: unyt.unyt_array, other: unyt.unyt_array):
        if isinstance(other, unit_system._Flow360BaseUnit):
            return flow360_unit_array_eq(other, self)
        if isinstance(self, unyt.unyt_quantity):
            return np.ndarray.__eq__(self, other)
        elif self.size == other.size:
            return all(self[i] == other[i] for i in range(len(self)))
        return False

    def unyt_array_ne(self: unyt.unyt_array, other: unyt.unyt_array):
        if isinstance(other, unit_system._Flow360BaseUnit):
            return flow360_unit_array_ne(other, self)
        if isinstance(self, unyt.unyt_quantity):
            return np.ndarray.__ne__(self, other)
        elif self.size == other.size:
            return any(self[i] != other[i] for i in range(len(self)))
        return True

    def flow360_unit_array_eq(
        self: unit_system._Flow360BaseUnit, other: unit_system._Flow360BaseUnit
    ):
        if isinstance(other, (unit_system._Flow360BaseUnit, unyt.unyt_array)):
            if self.size == other.size:
                if str(self.units) == str(other.units):
                    if self.size == 1:
                        return np.ndarray.__eq__(self.v, other.v)
                    if isinstance(other, unyt.unyt_array):
                        other = unit_system._Flow360BaseUnit.factory(other.v, str(other.units))
                    return np.all(np.ndarray.__eq__(v.v, o.v) for v, o in zip(self, other))
        return False

    def flow360_unit_array_ne(
        self: unit_system._Flow360BaseUnit, other: unit_system._Flow360BaseUnit
    ):
        if isinstance(other, (unit_system._Flow360BaseUnit, unyt.unyt_array)):
            if self.size == other.size:
                if str(self.units) == str(other.units):
                    if self.size == 1:
                        return np.ndarray.__ne__(self.v, other.v)
                    if isinstance(other, unyt.unyt_array):
                        other = unit_system._Flow360BaseUnit.factory(other.v, str(other.units))
                    return np.any(np.ndarray.__ne__(v.v, o.v) for v, o in zip(self, other))
        return True

    unyt.unyt_array.__eq__ = unyt_array_eq
    unyt.unyt_array.__ne__ = unyt_array_ne
    unit_system._Flow360BaseUnit.__eq__ = flow360_unit_array_eq
    unit_system._Flow360BaseUnit.__ne__ = flow360_unit_array_ne

    # Yield control to the test
    yield

    # Restore original methods
    unyt.unyt_array.__eq__ = original_unyt_eq
    unyt.unyt_array.__ne__ = original_unyt_ne
    unit_system._Flow360BaseUnit.__eq__ = original_flow360_eq
    unit_system._Flow360BaseUnit.__ne__ = original_flow360_ne
