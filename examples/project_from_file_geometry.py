import flow360 as fl
from flow360.component.project import Project
from flow360.examples import Airplane

fl.Env.dev.active()

Airplane.get_files()

project = Project.from_file(
    Airplane.geometry,
    name="airplane-geometry-python-upload",
    solver_version="workbench-24.9.3",
    tags=["python"],
)

geometry = project.geometry

print(geometry)
