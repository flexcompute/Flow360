import flow360 as fl
from flow360.component.geometry import Geometry
from flow360.examples import Cylinder3D

fl.Env.preprod.active()

geometry_draft = Geometry.from_file(
    Cylinder3D.geometry,
    name="testing-cylinder3d-geometry",
)
geometry = geometry_draft.submit()

print(geometry)
