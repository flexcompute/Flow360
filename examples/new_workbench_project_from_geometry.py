import flow360.component.v1.modules as fl
from flow360.component.geometry import Geometry
from flow360.examples import Airplane

fl.Env.dev.active()

Airplane.get_files()

# geometry
geometry = Geometry.from_file(
    Airplane.geometry,
    project_name="airplane-geometry-python-upload",
    solver_version="workbench-24.9.2",
    tags=["python"],
)

geometry = geometry.submit()

print(geometry)
