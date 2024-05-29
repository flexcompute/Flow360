import flow360 as fl
from flow360.component.geometry import Geometry
from flow360.examples import Octahedron

geometry = Geometry.from_file(Octahedron.geometry, name="octahedron")
geometry = geometry.submit()

print(geometry)
