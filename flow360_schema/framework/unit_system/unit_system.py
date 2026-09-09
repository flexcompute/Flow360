"""Unit system context manager and predefined systems.

This module requires ``unyt``. The optional-dependency guard lives in the
package ``__init__`` files that import from here — by the time this module
is loaded, unyt is assumed to be installed.
"""

from __future__ import annotations

import contextvars
import logging
from types import TracebackType
from typing import Any, Literal

import pydantic as pd
import unyt as u
import unyt.dimensions as udim  # type: ignore[import-untyped]
from sympy import Symbol  # type: ignore[import-untyped]

from flow360_schema.framework.unit_system.base_system_type import BaseSystemType
from flow360_schema.framework.validation.context import unit_system_manager

log = logging.getLogger(__name__)

# Token stored per-context (not per-instance) so concurrent tasks using the
# same singleton UnitSystem each track their own token independently.
_unit_system_token_ctx: contextvars.ContextVar[contextvars.Token[Any] | None] = contextvars.ContextVar(
    "_unit_system_token_ctx", default=None
)

# ---------------------------------------------------------------------------
# Register custom dimensions with unyt so that system lookups work
# ---------------------------------------------------------------------------
udim.viscosity = udim.pressure * udim.time
udim.kinematic_viscosity = udim.length * udim.length / udim.time
udim.angular_velocity = udim.angle / udim.time
udim.acceleration = udim.length / udim.time**2
udim.heat_flux = udim.mass / udim.time**3
udim.moment = udim.force * udim.length
udim.heat_source = udim.mass / udim.time**3 / udim.length
udim.specific_heat_capacity = udim.length**2 / udim.temperature / udim.time**2
udim.thermal_conductivity = udim.mass / udim.time**3 * udim.length / udim.temperature
udim.inverse_area = 1 / udim.area
udim.inverse_length = 1 / udim.length
udim.mass_flow_rate = udim.mass / udim.time
udim.specific_energy = udim.length**2 * udim.time ** (-2)
udim.frequency = udim.time ** (-1)
udim.delta_temperature = Symbol("(delta temperature)", positive=True)

# Configure delta_temperature and imperial temperature in unyt registries
u.unit_systems.imperial_unit_system["temperature"] = u.Unit("degF").expr
u.unit_systems.imperial_unit_system["delta_temperature"] = u.Unit("delta_degF").expr
u.unit_systems.mks_unit_system["delta_temperature"] = u.Unit("K").expr
u.unit_systems.cgs_unit_system["delta_temperature"] = u.Unit("K").expr

# ---------------------------------------------------------------------------
# Predefined unyt systems
# ---------------------------------------------------------------------------
_PREDEFINED_UNYT_SYSTEMS = {
    BaseSystemType.SI: u.unit_systems.mks_unit_system,
    BaseSystemType.CGS: u.unit_systems.cgs_unit_system,
    BaseSystemType.IMPERIAL: u.unit_systems.imperial_unit_system,
}

_BASE_DIM_NAMES = ("mass", "length", "time", "temperature", "angle")


def _to_quantity(val: Any) -> Any:
    """Normalize a unit value to unyt_quantity(1.0, unit)."""
    if isinstance(val, u.unyt_quantity):
        return val
    return u.unyt_quantity(1.0, val)


# ---------------------------------------------------------------------------
# UnitSystem
# ---------------------------------------------------------------------------
class UnitSystem:
    """Unit system defined by base physical dimensions.

    Derived dimensions (velocity, force, etc.) are computed from base units
    on demand via ``__getitem__``, delegating to the backing unyt system.
    """

    mass: Any
    length: Any
    time: Any
    temperature: Any
    angle: Any
    name: str
    _verbose: bool
    _base_system: BaseSystemType | None
    _unyt_system: Any

    def __init__(
        self,
        *,
        base_system: BaseSystemType | str | None = None,
        verbose: bool = True,
        mass: Any = None,
        length: Any = None,
        time: Any = None,
        temperature: Any = None,
        angle: Any = None,
    ) -> None:
        if base_system is not None:
            base_system = BaseSystemType(base_system)

        self.name = "Custom"
        self._verbose = verbose
        self._base_system = base_system

        # Resolve base units from predefined system or explicit args
        predefined = _PREDEFINED_UNYT_SYSTEMS.get(self._base_system) if self._base_system is not None else None
        explicit = {
            "mass": mass,
            "length": length,
            "time": time,
            "temperature": temperature,
            "angle": angle,
        }

        for dim in _BASE_DIM_NAMES:
            val = explicit[dim]
            if val is None and predefined is not None:
                val = predefined[dim]
            if val is None:
                raise ValueError(f"Missing base unit: {dim}")
            setattr(self, dim, _to_quantity(val))

        # Backing unyt system for derived dimension lookups
        if predefined is not None and all(v is None for v in explicit.values()):
            self._unyt_system = predefined
        else:
            # NOTE: u.UnitSystem() registers globally in unyt's registry.
            # Not idempotent — each call creates a new entry. Acceptable
            # because custom (non-predefined) systems are rarely created.
            self._unyt_system = u.UnitSystem(
                f"_flow360_{id(self)}",
                length_unit=self.length,
                mass_unit=self.mass,
                time_unit=self.time,
                temperature_unit=self.temperature,
                angle_unit=self.angle,
            )

    def __getitem__(self, item: str | Any) -> Any:
        """Lookup unit by dimension name or unyt dimension expression."""
        if isinstance(item, str) and item in _BASE_DIM_NAMES:
            return getattr(self, item)
        # Derived dimensions — delegate to unyt system
        if self._unyt_system is not None:
            result = self._unyt_system[item]
            return _to_quantity(result)
        raise KeyError(f"Cannot resolve dimension: {item}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UnitSystem):
            return NotImplemented
        return all(getattr(self, d) == getattr(other, d) for d in _BASE_DIM_NAMES)

    def system_repr(self) -> str:
        """(mass, length, time, temperature) string representation."""
        units = [
            str(unit.units if unit.v == 1.0 else unit) for unit in [self.mass, self.length, self.time, self.temperature]
        ]
        return f"({', '.join(units)})"

    def _assert_no_active_unit_system(self) -> None:
        active = unit_system_manager.current
        if active is None:
            return
        active_name = active.system_repr()
        raise RuntimeError(
            "Nested unit system context is not allowed. "
            f"Active unit system: {active_name}. "
            f"Attempted: {self.system_repr()}. "
            "Please remove the inner unit system context."
        )

    def __enter__(self) -> UnitSystem:
        self._assert_no_active_unit_system()
        if self._verbose:
            log.info("using: %s unit system for unit inference.", self.system_repr())
        token = unit_system_manager.set_current(self)
        _unit_system_token_ctx.set(token)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        token = _unit_system_token_ctx.get()
        if token is None:
            raise RuntimeError("Unit system context exit called without a matching enter.")
        unit_system_manager.reset_current(token)
        _unit_system_token_ctx.set(None)


# ---------------------------------------------------------------------------
# Predefined systems
# ---------------------------------------------------------------------------
class _PredefinedUnitSystem(UnitSystem):
    def system_repr(self) -> str:
        return self.name


class SIUnitSystem(_PredefinedUnitSystem):
    def __init__(self, verbose: bool = True) -> None:
        super().__init__(base_system=BaseSystemType.SI, verbose=verbose)
        self.name = "SI"


class CGSUnitSystem(_PredefinedUnitSystem):
    def __init__(self, verbose: bool = True) -> None:
        super().__init__(base_system=BaseSystemType.CGS, verbose=verbose)
        self.name = "CGS"


class ImperialUnitSystem(_PredefinedUnitSystem):
    def __init__(self, verbose: bool = True) -> None:
        super().__init__(base_system=BaseSystemType.IMPERIAL, verbose=verbose)
        self.name = "Imperial"


# ---------------------------------------------------------------------------
# Flow360 unit system factory
# ---------------------------------------------------------------------------
def create_flow360_unit_system(
    length: Any,
    velocity: Any,
    density: Any,
    temperature: Any,
) -> UnitSystem:
    """Create a Flow360 unit system from CFD reference quantities.

    Derives base dimensions: time = length / velocity, mass = density * length^3.
    Returns a regular UnitSystem instance.
    """
    length = _to_quantity(length)
    velocity = _to_quantity(velocity)
    density = _to_quantity(density)
    temperature = _to_quantity(temperature)
    return UnitSystem(
        mass=density * length**3,
        length=length,
        time=length / velocity,
        temperature=temperature,
        angle=u.unyt_quantity(1.0, "rad"),
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Type alias and module-level instances
# ---------------------------------------------------------------------------
UnitSystemType = UnitSystem

SI_unit_system = SIUnitSystem()
CGS_unit_system = CGSUnitSystem()
imperial_unit_system = ImperialUnitSystem()

# ---------------------------------------------------------------------------
# Unit system lookup table and serialization config
# ---------------------------------------------------------------------------
_UNIT_SYSTEMS: dict[str, UnitSystem] = {
    "SI": SI_unit_system,
    "CGS": CGS_unit_system,
    "Imperial": imperial_unit_system,
}


class UnitSystemConfig(pd.BaseModel):
    """Serialization-only model for unit system selection. JSON: {"name": "SI"}"""

    name: Literal["SI", "CGS", "Imperial"]

    def resolve(self) -> UnitSystem:
        """Return the corresponding UnitSystem instance."""
        return _UNIT_SYSTEMS[self.name]
