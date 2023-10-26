from __future__ import annotations
import pydantic as pd

import unyt



class UnitSystemManager:

    def __init__(self):
        """
        Initialize the UnitSystemManager.
        """
        self._current = None

    @property
    def current(self) -> UnitSystem:
        """
        Get the current UnitSystem.
        :return: UnitSystem
        """
        return self._current

    def copy_current(self):
        if self._current:
            return self._current.copy(deep=True)
        return None

    def set_current(self, unit_system: UnitSystem):
        """
        Set the current UnitSystem.
        :param config:
        :return:
        """
        self._current = unit_system


unit_system_manager = UnitSystemManager()


def _has_dimensions(quant, dim):
    """Checks the argument has the right dimensionality."""
    try:
        arg_dim = quant.units.dimensions
    except AttributeError:
        arg_dim = unyt.dimensionless
    return arg_dim == dim


class DimensionedType:
    """:class: DimensionedType"""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value):
        """validator for value"""
        if unit_system_manager.current:
            print('inside context')
            if isinstance(value, (int, float)):
                # this needs more elegant solution:
                return value * unit_system_manager.current[str(cls.dim)[1:-1]]

        if not _has_dimensions(value, cls.dim):
            raise TypeError(f"arg '{value}' does not match {cls.dim}")
        return value


class LengthType(DimensionedType):
    """:class: LengthType"""
    dim = unyt.dimensions.length


class MassType(DimensionedType):
    """:class: MassType"""
    dim = unyt.dimensions.mass


class TimeType(DimensionedType):
    """:class: TimeType"""
    dim = unyt.dimensions.time


class TemperatureType(DimensionedType):
    """:class: TimeType"""
    dim = unyt.dimensions.temperature


class VelocityType(DimensionedType):
    """:class: VelocityType"""
    dim = unyt.dimensions.velocity


class AreaType(DimensionedType):
    """:class: AreaType"""
    dim = unyt.dimensions.area


class ForceType(DimensionedType):
    """:class: ForceType"""
    dim = unyt.dimensions.force


class PressureType(DimensionedType):
    """:class: PressureType"""
    dim = unyt.dimensions.pressure


class DensityType(DimensionedType):
    """:class: DensityType"""
    dim = unyt.dimensions.density



# we probably want to merge this unit system with unyt unit system
class UnitSystem(pd.BaseModel):
    mass: MassType = pd.Field()
    length: LengthType = pd.Field()
    time: TimeType = pd.Field()
    temperature: TemperatureType = pd.Field()

    def __getitem__(self, item):
        """to support [] access"""
        return getattr(self, item)

    def __enter__(self):
        print(
            f"using: ({self.mass}, {self.length}, {self.time}, {self.temperature}) unit system")
        unit_system_manager.set_current(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("exiting unit system context")
        unit_system_manager.set_current(None)


_SI = unyt.unit_systems.mks_unit_system

SI_unit_system = UnitSystem(mass=_SI['mass'], 
                            length=_SI['length'],
                            time=_SI['time'],
                            temperature=_SI['temperature'])
