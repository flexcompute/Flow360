import flow360 as fl
from flow360.component.geometry import Geometry

fl.Env.dev.active()

geometry_file_path = "cylinder_v2.x_t"
geometry = Geometry.from_file(
    geometry_file_path, name="testing_geometry", solver_version="release-24.2.2"
)
geometry = geometry.submit()

print(geometry)
