import flow360 as fl
from flow360.component.geometry import Geometry
from flow360.examples import Cylinder

geometry_draft = Geometry.from_file(
    Cylinder.geometry,
    name="testing-cylinder-geometry",
)
geometry = geometry_draft.submit()

print(geometry)
