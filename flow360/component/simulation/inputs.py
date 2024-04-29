from flow360.component.simulation.base_model import Flow360BaseModel

"""
    - Different stages of the simulation that can either come from cloud or local files. As the simulation progresses, each of these will get populated/updated if not specified at the beginning. All these attributes should have methods to compute/update/retrieve the params. Only one/zero of them can be specified in the `Simulation` constructor.
"""


class _tempGeometryMeta:
    # To be implemented
    ...


class Geometry(Flow360BaseModel):

    geoemtry_meta: _tempGeometryMeta

    def from_file(self, filename):
        pass

    def from_cloud(self, id):
        pass


class SurfaceMesh(Flow360BaseModel):

    def from_file(self, filename):
        pass

    def from_cloud(self, id):
        pass


class VolumeMesh(Flow360BaseModel):

    def from_file(self, filename):
        pass

    def from_cloud(self, id):
        pass
