"""
This module provides functions for handling unit conversion into flow360 solver unit system.
"""

# pylint: disable=duplicate-code

import operator
from functools import reduce

from flow360.component.simulation.unit_system import u

from ...exceptions import Flow360ConfigurationError

LIQUID_IMAGINARY_FREESTREAM_MACH = 0.05


class RestrictedUnitSystem(u.UnitSystem):
    """UnitSystem that blocks conversions for unsupported base dimensions.

    Automatically derives supported dimensions from which unit arguments are
    provided. Missing base units get placeholder values internally but are
    masked so that conversion attempts raise ValueError.

    Examples::

        # Meshing mode: only length defined, velocity/mass/temperature blocked
        RestrictedUnitSystem("nondim", length_unit=0.5 * u.m)

        # Full mode: all units provided, no restrictions
        RestrictedUnitSystem("nondim", length_unit=..., mass_unit=...,
                             time_unit=..., temperature_unit=...)
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        name,
        length_unit,
        mass_unit=None,
        time_unit=None,
        temperature_unit=None,
        **kwargs,
    ):
        supported = {u.dimensions.length, u.dimensions.angle}
        if mass_unit is not None:
            supported.add(u.dimensions.mass)
        if time_unit is not None:
            supported.add(u.dimensions.time)
        if temperature_unit is not None:
            supported.add(u.dimensions.temperature)

        super().__init__(
            f"{name}_{id(self)}",
            length_unit=length_unit,
            mass_unit=mass_unit or 1 * u.kg,
            time_unit=time_unit or 1 * u.s,
            temperature_unit=temperature_unit or 1 * u.K,
            **kwargs,
        )

        # All 4 base dims provided — no restrictions
        if len(supported) == 4:
            self._supported_dims = None
            return

        # Mask unsupported base dimensions in units_map so that
        # get_base_equivalent's fast path doesn't bypass our check
        self._supported_dims = supported
        for dim in list(self.units_map.keys()):
            if not dim.free_symbols <= supported:
                self.units_map[dim] = None

    def __getitem__(self, key):
        if isinstance(key, str):
            key = getattr(u.dimensions, key)
        if self._supported_dims is not None:
            unsupported = key.free_symbols - self._supported_dims
            if unsupported:
                names = ", ".join(str(s) for s in unsupported)
                raise ValueError(
                    f"Cannot non-dimensionalize {key}: "
                    f"base units for {names} are not defined in this context."
                )
        return super().__getitem__(key)


def get_from_dict_by_key_list(key_list, data_dict):
    """
    Get a value from a nested dictionary using a list of keys.

    Parameters
    ----------
    key_list : List[str]
        List of keys specifying the path to the desired value.
    data_dict : dict
        The dictionary from which to retrieve the value.

    Returns
    -------
    value
        The value located at the specified nested path in the dictionary.
    """

    return reduce(operator.getitem, key_list, data_dict)


def need_conversion(value):
    """
    Check if a value needs conversion to flow360 units.

    Parameters
    ----------
    value : Any
        The value to check for conversion.

    Returns
    -------
    bool
        True if conversion is needed (i.e. value carries physical units), False otherwise.
    """

    return hasattr(value, "units")


def require(required_parameter, required_by, params):
    """
    Ensure that required parameters are present in the provided dictionary.

    Parameters
    ----------
    required_parameter : List[str]
        List of keys specifying the path to the desired parameter that is required.
    required_by : List[str]
        List of keys specifying the path to the parameter that requires required_parameter for unit conversion.
    params : SimulationParams
        The dictionary containing the parameters.

    Raises
    ------
    Flow360ConfigurationError
        Configuration error due to missing parameter.
    """

    required_msg = f'required by {" -> ".join(required_by)} for unit conversion'
    try:
        params_as_dict = params
        if not isinstance(params_as_dict, dict):
            params_as_dict = params.model_dump()
        value = get_from_dict_by_key_list(required_parameter, params_as_dict)
        if value is None:
            raise ValueError

    except Exception as err:
        raise Flow360ConfigurationError(
            message=f'{" -> ".join(required_parameter)} is {required_msg}.',
            field=required_by,
            dependency=required_parameter,
        ) from err


def get_flow360_unit_system_liquid(params, to_flow360_unit: bool = False) -> u.UnitSystem:
    """
    Returns the flow360 unit system when liquid operating condition is used.

    Parameters
    ----------
    params : SimulationParams
        The parameters needed for unit conversion that uses liquid operating condition.
    to_flow360_unit : bool, optional
        Whether we want user input to be converted to flow360 unit system.
        The reverse path requires different conversion logic (from solver output to non-flow360 unit system)
        since the solver output is already re-normalized by `reference velocity` due to "velocityScale".

    Returns
    -------
    u.UnitSystem
        The flow360 unit system.

    ##-- When to_flow360_unit is True,
    ##-- time unit should be changed such that it takes into consideration
    ##-- the fact that solver output already multiplied by "velocityScale"
    """

    if to_flow360_unit:
        base_velocity = params.base_velocity
    else:
        base_velocity = params.reference_velocity  # pylint:disable=protected-access

    time_unit = params.base_length / base_velocity
    return u.UnitSystem(
        name="flow360_liquid",
        length_unit=params.base_length,
        mass_unit=params.base_mass,
        time_unit=time_unit,
        temperature_unit=params.base_temperature,
    )


def compute_udf_dimensionalization_factor(params, requested_unit, using_liquid_op):
    """

    Returns the dimensionalization coefficient and factor given a requested unit

    Parameters
    ----------
    params : SimulationParams
        The parameters needed for unit conversion.
    unit: u.Unit
        The unit to compute the factors.
    using_liquid_op : bool
        If True, compute the factor based on the flow360_liquid unit system.
    Returns
    -------
    coefficient and offset for unit conversion from the requested unit to flow360 unit

    """

    def _compute_coefficient_and_offset(source_unit: u.Unit, target_unit: u.Unit):
        y2 = (2.0 * target_unit).in_units(source_unit).value
        y1 = (1.0 * target_unit).in_units(source_unit).value
        x2 = 2.0
        x1 = 1.0

        coefficient = (y2 - y1) / (x2 - x1)
        offset = y1 / coefficient - x1

        return coefficient, offset

    flow360_unit_system = (
        params.flow360_unit_system
        if not using_liquid_op
        else get_flow360_unit_system_liquid(params=params)
    )
    # Note: Effectively assuming that all the solver vars uses radians and also the expressions expect radians
    flow360_unit_system["angle"] = u.rad  # pylint:disable=no-member
    flow360_unit = flow360_unit_system[requested_unit.dimensions]
    coefficient, offset = _compute_coefficient_and_offset(
        source_unit=requested_unit, target_unit=flow360_unit
    )
    return coefficient, offset
