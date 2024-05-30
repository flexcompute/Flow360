import flow360 as fl
from flow360.component.geometry import Geometry
from flow360.examples import Airplane

fl.Env.dev.active()

geometry = Geometry.from_file(
    Airplane.geometry, name="testing-airplane-csm-geometry",
)
geometry = geometry.submit()

print(geometry)
