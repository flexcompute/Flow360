import pytest

from flow360.component.meshing.params import (
    Aniso,
    BoxRefinement,
    CylinderRefinement,
    Edges,
    Face,
    FaceInVolume,
    Faces,
    Farfield,
    ProjectAniso,
    RotorDisk,
    SurfaceMeshingParams,
    Volume,
    VolumeMeshingParams,
)

from .utils import compare_to_ref, to_file_from_file_test


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_edges():
    edges = Edges.parse_raw(
        """
        {
            "leadingEdge":  {
                "type": "aniso",
                "method": "angle",
                "value": 1
            },
            "trailingEdge":  {
                "type": "aniso",
                "method": "height",
                "value": 1e-3
            },
            "hubCircle": {
                "type": "aniso",
                "method": "height",
                "value": 0.1
            },
            "hubSplitEdge": {
                "type": "projectAnisoSpacing"
            }
        }
        """
    )
    assert edges
    to_file_from_file_test(edges)

    edges = Edges(
        leadingEdge={"type": "aniso", "method": "angle", "value": 1},
        trailingEdge={"type": "aniso", "method": "height", "value": 1e-3},
        hubCircle={"type": "aniso", "method": "height", "value": 0.1},
        hubSplitEdge={"type": "projectAnisoSpacing"},
    )

    assert edges
    to_file_from_file_test(edges)

    edges = Edges(
        leadingEdge=Aniso(method="angle", value=1),
        trailingEdge=Aniso(method="height", value=1e-3),
        hubCircle=Aniso(method="height", value=0.1),
        hubSplitEdge=ProjectAniso(),
    )

    assert edges
    to_file_from_file_test(edges)

    with pytest.raises(ValueError):
        edges = Edges.parse_raw(
            """
            {
                "leadingEdge":  {
                    "type": "unsupported",
                    "method": "angle",
                    "value": 1
                }
            }
            """
        )


def test_faces():
    faces = Faces.parse_raw(
        """
        {
            "rightWing": {
                "maxEdgeLength": 0.05
            },
            "fuselage": {
                "maxEdgeLength": 0.05
            }
        }
        """
    )
    assert faces
    to_file_from_file_test(faces)

    with pytest.raises(ValueError):
        faces = Faces.parse_raw(
            """
            {
                "rightWing": {
                    "maxEdgeLength": 0.05,
                    "otherParams": 2
                }
            }
            """
        )


def test_surface_meshing_params():
    params = SurfaceMeshingParams.parse_raw(
        """
        {
            "maxEdgeLength": 0.1,
            "curvatureResolutionAngle": 15,
            "growthRate": 1.2,
            "faces": {
                "mysphere": {
                    "maxEdgeLength": 0.05
                }
            },
            "edges": {
                "leadingEdge":  {
                    "type": "aniso",
                    "method": "angle",
                    "value": 1
                }
            }
        }
    """
    )
    to_file_from_file_test(params)

    params = SurfaceMeshingParams(
        max_edge_length=0.1,
        faces={"mysphere": Face(max_edge_length=0.05)},
        edges={"leadingEdge": Aniso(method="angle", value=1)},
    )
    to_file_from_file_test(params)

    assert params.flow360_json()

    compare_to_ref(params, "ref/meshing_params/ref.json")
    compare_to_ref(params, "ref/meshing_params/ref.yaml")


def test_volume_meshing_params_from_obj():
    params = VolumeMeshingParams.parse_obj(
        {
            "volume": {"firstLayerThickness": 1e-3},
            "refinement": [
                {
                    "size": [4, 3, 2],
                    "center": [2, 0, 0],
                    "spacing": 0.05,
                    "axisOfRotation": [0, 0, 1],
                    "angleOfRotation": 45,
                },
                {
                    "type": "cylinder",
                    "radius": 4,
                    "length": 5,
                    "center": [5, 0, 0],
                    "spacing": 0.05,
                    "axis": [1, 0, 0],
                },
            ],
        }
    )
    assert params
    assert isinstance(params.refinement[0], BoxRefinement)


def test_volume_meshing_sliding_interfaces():
    params = VolumeMeshingParams.parse_obj(
        {
            "volume": {"firstLayerThickness": 1e-3},
            "slidingInterfaces": [
                {
                    "name": "inner",
                    "innerRadius": 0,
                    "outerRadius": 0.75,
                    "thickness": 0.5,
                    "axisOfRotation": [0, 0, 1],
                    "center": [0, 0, 0],
                    "spacingAxial": 0.2,
                    "spacingRadial": 0.2,
                    "spacingCircumferential": 0.2,
                    "enclosedObjects": ["hub", "blade1", "blade2", "blade3"],
                },
                {
                    "name": "mid",
                    "innerRadius": 0,
                    "outerRadius": 2.0,
                    "thickness": 2.0,
                    "axisOfRotation": [0, 1, 0],
                    "center": [0, 0, 0],
                    "spacingAxial": 0.2,
                    "spacingRadial": 0.2,
                    "spacingCircumferential": 0.2,
                    "enclosedObjects": ["slidingInterface-inner"],
                },
                {
                    "innerRadius": 0,
                    "outerRadius": 2.0,
                    "thickness": 2.0,
                    "axisOfRotation": [0, 1, 0],
                    "center": [0, 5, 0],
                    "spacingAxial": 0.2,
                    "spacingRadial": 0.2,
                    "spacingCircumferential": 0.2,
                    "enclosedObjects": ["rotorDisk-enclosed"],
                },
                {
                    "innerRadius": 1.5,
                    "outerRadius": 2.0,
                    "thickness": 2.0,
                    "axisOfRotation": [0, 1, 0],
                    "center": [0, -5, 0],
                    "spacingAxial": 0.2,
                    "spacingRadial": 0.2,
                    "spacingCircumferential": 0.2,
                },
                {
                    "name": "outer",
                    "innerRadius": 0,
                    "outerRadius": 8,
                    "thickness": 6,
                    "axisOfRotation": [1, 0, 0],
                    "center": [0, 0, 0],
                    "spacingAxial": 0.4,
                    "spacingRadial": 0.4,
                    "spacingCircumferential": 0.4,
                    "enclosedObjects": [
                        "slidingInterface-mid",
                        "rotorDisk-1",
                        "slidingInterface-2",
                        "slidingInterface-3",
                    ],
                },
            ],
        }
    )
    print(params)


def test_volume_meshing_params():
    box_refinement_left = BoxRefinement(
        center=(3.6, -5, 0),
        axis_of_rotation=(0.06052275, -0.96836405, 0),
        angle_of_rotation=76,
        size=(2, 2, 0.5),
        spacing=0.1,
    )
    box_refinement_right = box_refinement_left.copy(update={"center": (3.6, 5, 0)})

    rotor_disk_left = RotorDisk(
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

    params = VolumeMeshingParams(
        refinement_factor=1,
        volume=Volume(first_layer_thickness=1e-5),
        refinement=[
            box_refinement_left,
            box_refinement_right,
            BoxRefinement(center=(10, 0, 0), size=(20, 15, 10), spacing=1),
            CylinderRefinement(
                radius=0.75, length=11, spacing=0.2, axis=(1, 0, 0), center=(5, 0, 0)
            ),
        ],
        rotor_disks=[rotor_disk_left, rotor_disk_right],
    )
    assert params


def test_volume_meshing_params_farfield():
    params = VolumeMeshingParams(
        volume=Volume(first_layer_thickness=1e-5), farfield=Farfield(type="auto")
    )
    params = VolumeMeshingParams(
        volume=Volume(first_layer_thickness=1e-5), farfield=Farfield(type="quasi-3d")
    )
    params = VolumeMeshingParams(
        volume=Volume(first_layer_thickness=1e-5), farfield=Farfield(type="user-defined")
    )


def test_volume_meshing_face_in_volume():
    params = VolumeMeshingParams(
        volume=Volume(first_layer_thickness=1e-5),
        faces={
            "floor": FaceInVolume(type="aniso", first_layer_thickness=1e-5),
            "ceiling": FaceInVolume(type="none"),
            "adjacent2floor": FaceInVolume(type="projectAnisoSpacing"),
        },
    )
