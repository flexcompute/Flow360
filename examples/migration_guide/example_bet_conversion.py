from examples.migration_guide.bet_disk_converter import bet_disk_convert
from flow360.component.simulation.unit_system import u
from flow360.component.simulation.units import flow360_angular_velocity_unit

BETDisks, Cylinders = bet_disk_convert(
    file="BET_tutorial_Flow360.json",
    save=True,
    length_unit=u.m,
    omega_unit=flow360_angular_velocity_unit,
)

print(BETDisks[0])

"""
with SI_unit_system:
    params = fl.SimulationParams(
        ...
        models=[
            ...
            BETDisk(BETDisks[0]),
            ...
        ]
        ...
    )
"""
