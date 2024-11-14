import json

from examples.migration_guide.bet_disk import bet_disk_convert
from flow360.component.simulation.unit_system import u
from flow360.component.simulation.units import flow360_angular_velocity_unit
from flow360.examples import TutorialBETDisk

BETDisks, Cylinders = bet_disk_convert(
    file=TutorialBETDisk.case_json,
    save=True,
    length_unit=u.m,
    omega_unit=flow360_angular_velocity_unit,
)

print(json.dumps(BETDisks[0].model_dump(), indent=4))
