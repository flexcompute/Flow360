import unyt as u

""" 
Extra units to be included in the dimensioned type schema 
(default SI, CGS, imperial units are included by default)
"""
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
    "angular_velocity": [],
}
