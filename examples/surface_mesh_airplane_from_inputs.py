import flow360 as fl
from flow360.examples import Airplane

params = fl.SurfaceMeshingParams(
    max_edge_length=0.16,
    edges={
        "leadingEdge": fl.meshing.Aniso(method="angle", value=5),
        "trailingEdge": fl.meshing.Aniso(method="height", value=0.001),
        "root": fl.meshing.Aniso(method="aspectRatio", value=10),
        "tip": fl.meshing.UseAdjacent(),
        "fuselageSplit": fl.meshing.ProjectAniso(),
    },
    faces={
        "rightWing": fl.meshing.Face(max_edge_length=0.08),
        "leftWing": fl.meshing.Face(max_edge_length=0.08),
        "fuselage": fl.meshing.Face(max_edge_length=0.1),
    },
)

surface_mesh = fl.SurfaceMesh.create(
    Airplane.geometry, params=params, name="airplane-new-python-client"
)
surface_mesh = surface_mesh.submit()

print(surface_mesh)
print(surface_mesh.params)
