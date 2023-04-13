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

surface_mesh = fl.SurfaceMesh.new(
    Airplane.geometry, params=params, name="airplane-new-python-client"
)
surface_mesh = surface_mesh.submit()


box_refinement_left = fl.meshing.BoxRefinement(
    center=(3.6, -5, 0),
    axis_of_rotation=(0.06052275, -0.96836405, 0),
    angle_of_rotation=76,
    size=(2, 2, 0.5),
    spacing=0.1,
)
box_refinement_right = box_refinement_left.copy(update={"center": (3.6, 5, 0)})

rotor_disk_left = fl.meshing.RotorDisk(
    innerRadius=0.0,
    outerRadius=2,
    thickness=0.42,
    axisThrust=(-0.96836405, -0.06052275, 0.24209101),
    center=(3.6, -5, 0),
    spacingAxial=0.03,
    spacingRadial=0.09,
    spacingCircumferential=0.09,
)
rotor_disk_right = rotor_disk_left.copy(update={"center": (3.6, 5, 0)})


params = fl.VolumeMeshingParams(
    volume=fl.meshing.Volume(first_layer_thickness=1e-5),
    refinement=[
        box_refinement_left,
        box_refinement_right,
        fl.meshing.BoxRefinement(center=(10, 0, 0), size=(20, 15, 10), spacing=1),
        fl.meshing.CylinderRefinement(
            radius=0.75, length=11, spacing=0.2, axis=(1, 0, 0), center=(5, 0, 0)
        ),
    ],
    rotor_disks=[rotor_disk_left, rotor_disk_right],
)

volume_mesh = surface_mesh.new_volume_mesh("airplane-new-python-client", params=params)
volume_mesh = volume_mesh.submit()
