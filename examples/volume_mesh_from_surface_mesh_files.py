from flow360.component.v1.meshing.params import SurfaceMeshingParams
from flow360.component.v1.meshing.params import VolumeMeshingParams
from flow360.examples import Airplane
from flow360.component.surface_mesh import SurfaceMesh

params = SurfaceMeshingParams(Airplane.surface_json)
surface_mesh = SurfaceMesh.create(
    Airplane.geometry, params=params, name="airplane-new-python-client"
)
surface_mesh = surface_mesh.submit()


params = VolumeMeshingParams(Airplane.volume_json)
volume_mesh = surface_mesh.create_volume_mesh("airplane-new-python-client", params=params)
volume_mesh = volume_mesh.submit()
