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
    "delta_temperature": [],
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

supported_units_by_front_end = {
    "(mass)": {"SI": "kg", "CGS": "g", "Imperial": "lb"},  # Not used in SimulationParams
    "(length)*(mass)/(time)**2": {
        "SI": "N",
        "CGS": "dyn",
        "Imperial": "lbf",
    },  # [Force] Currently not used in SimulationParams.
    "(length)**2*(mass)/(time)**3": {
        "SI": "W",
        "CGS": "erg/s",
        "Imperial": "hp",
    },  # [Power] Currently not used in SimulationParams.
    # Note: (length)**2*(mass)/(time)**2 can be both energy and moment.
    # Note: Using force*length for less user confusion.
    "(length)**2*(mass)/(time)**2": {
        "SI": "N*m",
        "CGS": "dyn*cm",
        "Imperial": "ft*lbf",
    },  # [Moment/Energy] Currently not used in SimulationParams.
    "(length)*(mass)/((temperature)*(time)**3)": {
        "SI": "W/m/K",
        "CGS": "erg/s/cm/K",
        "Imperial": "BTU/hr/ft/degF",
    },  # [Thermal conductivity] Not used in SimulationParams currently
    "(length)": {"SI": "m", "CGS": "cm", "Imperial": "ft"},
    "1/(length)": {"SI": "1/m", "CGS": "1/cm", "Imperial": "1/ft"},
    "(length)**2": {"SI": "m**2", "CGS": "cm**2", "Imperial": "ft**2"},
    "(length)**(-2)": {"SI": "1/m**2", "CGS": "1/cm**2", "Imperial": "1/ft**2"},
    "(length)/(time)": {"SI": "m/s", "CGS": "cm/s", "Imperial": "ft/s"},
    "(angle)/(time)": ["rad/s", "degree/s", "rpm"],  # list --> Unit system agnostic dimensions
    "(angle)": ["degree", "rad"],
    "(temperature)": {"SI": "K", "CGS": "K", "Imperial": "degF"},
    "(temperature_difference)": {"SI": "K", "CGS": "K", "Imperial": "delta_degF"},
    "(mass)/(time)": {"SI": "kg/s", "CGS": "g/s", "Imperial": "lb/s"},
    "(mass)/((length)*(time)**2)": {"SI": "Pa", "CGS": "dyn/cm**2", "Imperial": "lbf/ft**2"},
    "(mass)/(time)**3": {"SI": "kg/s**3", "CGS": "g/s**3", "Imperial": "lb/s**3"},
    "(mass)/((length)*(time)**3)": {
        "SI": "kg/(m*s**3)",
        "CGS": "g/(cm*s**3)",
        "Imperial": "lb/(ft*s**3)",
    },
    "(mass)/(length)**3": {"SI": "kg/m**3", "CGS": "g/cm**3", "Imperial": "lb/ft**3"},
    "(length)**2/((temperature)*(time)**2)": {
        "SI": "m**2/(K*s**2)",
        "CGS": "cm**2/(K*s**2)",
        "Imperial": "ft**2/(degF*s**2)",
    },
    "(mass)/((length)*(time))": {"SI": "Pa*s", "CGS": "dyn*s/cm**2", "Imperial": "lbf*s/ft**2"},
    "(length)**2/(time)**2": {"SI": "J/kg", "CGS": "erg/g", "Imperial": "lbf*ft/lb"},
    "1/(time)": ["Hz"],
    "(time)": ["s"],
    "(length)**2/(time)": {
        "SI": "m**2/s",
        "CGS": "cm**2/s",
        "Imperial": "ft**2/s",
    },  # Kinematic viscosity group.
}
