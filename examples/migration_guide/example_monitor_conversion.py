import os

from flow360.component.simulation.migration import ProbeOutput
from flow360.component.simulation.unit_system import flow360_length_unit

# Get the absolute path to the script file
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory to the script directory
os.chdir(script_dir)

my_monitor = ProbeOutput.read_all_v0_monitors(
    file_path="./ProbeOutput_tutorial_Flow360.json", length_unit=flow360_length_unit
)

print(my_monitor)

"""
with SI_unit_system:
    params = fl.SimulationParams(
        ...
        outputs=[
            ...
            *my_monitor,
            ...
        ]
        ...
    )
"""
