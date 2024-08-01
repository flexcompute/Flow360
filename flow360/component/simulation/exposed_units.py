"""
Extra units to be included in the dimensioned type schema
(default SI, CGS, imperial units are included by default)
"""

# pylint: disable=no-member
import unyt as u

# pylint: disable=duplicate-code

extra_units = {
    "mass": [],
    "length": [u.mm, u.inch],
    "time": [],
    "temperature": [],
    "velocity": [],
    "area": [],
    "force": [],
    "pressure": [],
    "density": [],
    "viscosity": [],
    "angular_velocity": [u.rpm],
    "heat_flux": [],
    "heat_source": [],
    "specific_heat_capacity": [],
    "thermal_conductivity": [],
    "inverse_area": [],
    "inverse_length": [],
    "angle": [],
    "specific_energy": [],
    "frequency": [],
    "mass_flow_rate": [],
    "power": [],
    "moment": [],
}

## In case we want to dictate the ordering in schema (for prettier presentation)
ordered_complete_units = {
    "length": [u.m, u.cm, u.mm, u.ft, u.inch],
    "temperature": [u.K, "degC", "degF", u.R],  # Cannot use u.degC since it __str__() to "\u00b0C"
}
