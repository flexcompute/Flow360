import flow360 as fl
from flow360.component.geometry import Geometry
from flow360.examples import CylinderGeometry

fl.Env.dev.active()

geometry = Geometry.from_file(
    CylinderGeometry.geometry, name="testing-cylinder-geometry", solver_version="release-24.2.2"
)
geometry = geometry.submit()

print(geometry)
