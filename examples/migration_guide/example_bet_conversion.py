import os

from flow360.component.simulation.migration import BETDisk
from flow360.component.simulation.unit_system import u

# Get the absolute path to the script file
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory to the script directory
os.chdir(script_dir)

BETDisk = BETDisk.read_single_v1_BETDisk(
    file_path="./BET_tutorial_Flow360.json",
    mesh_unit=u.m,
    time_unit=u.s,
)

print(BETDisk)

"""
with SI_unit_system:
    params = fl.SimulationParams(
        ...
        models=[
            ...
            BETDisk(BETDisk),
            ...
        ]
        ...
    )
"""
