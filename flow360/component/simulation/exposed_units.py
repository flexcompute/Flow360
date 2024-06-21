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
}
