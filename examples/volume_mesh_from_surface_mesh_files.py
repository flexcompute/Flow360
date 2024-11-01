import flow360.component.v1.modules as fl
from flow360.examples import Airplane

params = fl.SurfaceMeshingParams(Airplane.surface_json)
surface_mesh = fl.SurfaceMesh.create(
    Airplane.geometry, params=params, name="airplane-new-python-client"
)
surface_mesh = surface_mesh.submit()


params = fl.VolumeMeshingParams(Airplane.volume_json)
volume_mesh = surface_mesh.create_volume_mesh("airplane-new-python-client", params=params)
volume_mesh = volume_mesh.submit()
