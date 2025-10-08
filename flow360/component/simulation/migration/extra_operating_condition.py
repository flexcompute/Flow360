"""This module contains extra operating conditions."""

from typing import Optional

import pydantic as pd

import flow360.component.simulation.units as u
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
    Air,
    ThermalState,
)
from flow360.component.simulation.unit_system import (
    AbsoluteTemperatureType,
    AngleType,
    LengthType,
)
from flow360.log import log


# pylint: disable=too-many-arguments, no-member, duplicate-code
@pd.validate_call
def operating_condition_from_mach_muref(
    mach: pd.NonNegativeFloat,
    mu_ref: pd.PositiveFloat,
    project_length_unit: LengthType.Positive = pd.Field(
        description="The Length unit of the project."
    ),
    temperature: AbsoluteTemperatureType = 288.15 * u.K,
    alpha: Optional[AngleType] = 0 * u.deg,
    beta: Optional[AngleType] = 0 * u.deg,
    reference_mach: Optional[pd.PositiveFloat] = None,
) -> AerospaceCondition:
    """
    Create an `AerospaceCondition` from Mach number and reference dynamic viscosity.

    This function computes the thermal state based on the given Mach number,
    reference dynamic viscosity, and temperature, and returns an `AerospaceCondition` object
    initialized with the computed thermal state and given aerodynamic angles.

    Parameters
    ----------
    mach : NonNegativeFloat
        Freestream Mach number (must be non-negative).
    muRef : PositiveFloat
        Freestream reference dynamic viscosity defined with mesh unit (must be positive).
    project_length_unit: LengthType.Positive
        Project length unit.
    temperature : TemperatureType.Positive, optional
        Freestream static temperature (must be a positive temperature value). Default is 288.15 Kelvin.
    alpha : AngleType, optional
        Angle of attack. Default is 0 degrees.
    beta : AngleType, optional
        Sideslip angle. Default is 0 degrees.
    reference_mach : PositiveFloat, optional
        Reference Mach number. Default is None.

    Returns
    -------
    AerospaceCondition
        An `AerospaceCondition` object initialized with the given parameters.

    Raises
    ------
    ValidationError
        If the input values do not meet the specified constraints.
    ValueError
        If required parameters are missing or calculations cannot be performed.

    Examples
    --------
    Example usage:

    >>> condition = operating_condition_from_mach_muref(
    ...     mach=0.85,
    ...     mu_ref=4.291e-8,
    ...     project_length_unit=1 * u.mm,
    ...     temperature=288.15 * u.K,
    ...     alpha=2.0 * u.deg,
    ...     beta=0.0 * u.deg,
    ...     reference_mach=0.85,
    ... )
    >>> print(condition)
    AerospaceCondition(...)

    """

    if temperature == 288.15 * u.K:
        log.info("Default value of 288.15 K will be used as temperature.")

    material = Air()

    density = material.get_dynamic_viscosity(temperature) / (
        mu_ref * material.get_speed_of_sound(temperature) * project_length_unit
    )

    thermal_state = ThermalState(temperature=temperature, density=density)

    log.info(
        """Density and viscosity were calculated based on input data, ThermalState will be automatically created."""
    )

    # pylint: disable=no-value-for-parameter
    return AerospaceCondition.from_mach(
        mach=mach,
        alpha=alpha,
        beta=beta,
        thermal_state=thermal_state,
        reference_mach=reference_mach,
    )
