import numpy as np
import unyt as u

from flow360.component.types import Axis


def convert_unyt_object(data, unit_system):
    """
    Convert unyt object to plain numerical value in Flow360 unit system.
    """
    converted = data.in_base(unit_system=unit_system)
    if isinstance(converted, u.unyt_array):
        if converted.ndim == 0:
            return float(converted.value)
        return converted.value.tolist()
    return float(converted.value)


def convert_and_strip_units_inplace(data, unit_system: u.UnitSystem, legacy_mode: bool = False):
    """Convert flow-style {'value', 'units'} objects to plain numbers/lists.

    This function walks the nested dict/list structure and:
    - For dicts that contain both 'value' and 'units', converts 'value' into the base of
      the given unit system and returns the converted bare value (not wrapped in a dict).
    - For dicts that contain only 'value' (and optional metadata like 'typeName'), unwraps
      to the underlying value after recursively processing it.
    - For lists/tuples/unyt objects, converts elements and returns plain Python types.

    If a unit string cannot be parsed by unyt, the entry is left unchanged.

    For legacy_mode, the function will return the converted value as a dict with the key 'value' and the value will be the converted value.
    """

    def _convert_value(original_value, units_str):
        try:
            old_unit = u.Unit(units_str)
        except Exception:
            return None  # signal to skip conversion

        if isinstance(original_value, (list, tuple, np.ndarray)):
            quantity = np.asarray(original_value, dtype=np.float64) * old_unit
        else:
            quantity = np.float64(original_value) * old_unit

        converted = quantity.in_base(unit_system=unit_system)

        # Normalize to plain Python types
        if isinstance(converted, u.unyt_array):
            # TODO: Use the func
            if converted.ndim == 0:
                return float(converted.value)
            return converted.value.tolist()
        return float(converted.value)

    # Directly handle unyt objects
    if isinstance(data, (u.unyt_quantity, u.unyt_array)):
        return convert_unyt_object(data, unit_system)

    if isinstance(data, dict):
        has_value_key = "value" in data
        has_units_key = "units" in data

        # {'value': X, 'units': '...'} → convert and return bare X
        if has_value_key and has_units_key:
            if legacy_mode:
                data["value"] = _convert_value(data["value"], data["units"])
                data["units"] = "flow360_unit"
            else:
                return _convert_value(data["value"], data["units"])

        # Generic dict: recurse into items
        for key in list(data.keys()):
            data[key] = convert_and_strip_units_inplace(
                data[key], unit_system, legacy_mode=legacy_mode
            )
        return data

    if isinstance(data, list):
        for idx in range(len(data)):
            data[idx] = convert_and_strip_units_inplace(
                data[idx], unit_system, legacy_mode=legacy_mode
            )
        return data

    # Some dedicated handling for specific types
    if isinstance(data, Axis):
        return [float(x) for x in data]

    if isinstance(data, np.float64):
        return float(data)

    return data
