import flow360 as fl
from flow360.component.geometry import Geometry
from flow360.examples import CylinderGeometry

fl.Env.dev.active()

geometry = Geometry.from_file(
    CylinderGeometry.geometry, name="testing-cylinder-geometry",
)
geometry = geometry.submit()

print(geometry)
