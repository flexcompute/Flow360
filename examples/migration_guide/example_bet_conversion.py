import os

import flow360 as fl

from flow360.component.simulation.migration import BETDisk
from flow360.component.simulation.unit_system import u

# Get the absolute path to the script file
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory to the script directory
os.chdir(script_dir)

with fl.SI_unit_system:
    params = fl.SimulationParams(
        operating_condition=fl.AerospaceCondition(
            velocity_magnitude=10
        )
    )

my_BETDisk = BETDisk.read_single_v1_BETDisk(
    file_path="./BET_tutorial_Flow360.json",
    mesh_unit=u.m,
    freestream_temperature=params.operating_condition.thermal_state.temperature
)

print(my_BETDisk.omega)

# Converted BETDisk can be used in a simulation
"""
with SI_unit_system:
    params = fl.SimulationParams(
        ...
        models=[
            ...
            my_BETDisk,
            ...
        ]
        ...
    )
"""

# After creating params, changing the value of omega can be done by doing the following
"""
params.models[X].omega = 500 * fl.u.rpm

where X is an int specifying chosen BETDisk's position in the list of models
"""
