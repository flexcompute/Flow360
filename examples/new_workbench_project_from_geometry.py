import flow360 as fl
from flow360.component.geometry import Geometry

fl.Env.dev.active()

# geometry
geometry_draft = Geometry.from_file(
    "data/airplane_geometry.egads",
    project_name="airplane-geometry-python-upload",
    solver_version="workbench-24.9.2",
    tags=["python"],
)

geometry = geometry_draft.submit()
print(geometry)
