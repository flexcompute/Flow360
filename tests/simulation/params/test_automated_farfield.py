import os
import re

import pytest
import unyt as u

from flow360.component.geometry import Geometry, GeometryMeta
from flow360.component.project_utils import set_up_params_for_uploading
from flow360.component.resource_base import local_metadata_builder
from flow360.component.simulation import services
from flow360.component.simulation.meshing_param.face_params import SurfaceRefinement
from flow360.component.simulation.meshing_param.params import (
    MeshingDefaults,
    MeshingParams,
)
from flow360.component.simulation.meshing_param.volume_params import (
    AutomatedFarfield,
    UserDefinedFarfield,
)
from flow360.component.simulation.models.surface_models import (
    Freestream,
    SlipWall,
    SymmetryPlane,
    Wall,
)
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
)
from flow360.component.simulation.outputs.outputs import (
    SurfaceIntegralOutput,
    SurfaceOutput,
    UserDefinedField,
)
from flow360.component.simulation.primitives import Surface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import SI_unit_system
from flow360.component.surface_mesh_v2 import SurfaceMeshMetaV2, SurfaceMeshV2


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


@pytest.fixture()
def surface_mesh():
    sm = SurfaceMeshV2.from_local_storage(
        local_storage_path="data/surface_mesh",
        meta_data=SurfaceMeshMetaV2(
            **local_metadata_builder(
                id="aaa",
                name="aaa",
                cloud_path_prefix="aaa",
            )
        ),
    )
    return sm


def _run_validation(params, surface_mesh_obj, use_beta_mesher=True, use_geometry_AI=False):
    params = set_up_params_for_uploading(
        params=params,
        root_asset=surface_mesh_obj,
        length_unit=1 * u.m,
        use_beta_mesher=use_beta_mesher,
        use_geometry_AI=use_geometry_AI,
    )

    _, errors, _ = services.validate_model(
        params_as_dict=params.model_dump(exclude_none=True),
        validated_by=services.ValidationCalledBy.LOCAL,
        root_item_type="SurfaceMesh",
        validation_level="All",
    )
    return errors


def test_automated_farfield_surface_usage():
    # Test use of GhostSurface in meshing
    with pytest.raises(
        ValueError,
        match=re.escape("Can not find any valid entity of type ['Surface'] from the input."),
    ):
        with SI_unit_system:
            my_farfield = AutomatedFarfield(name="my_farfield")
            _ = SimulationParams(
                meshing=MeshingParams(
                    volume_zones=[
                        my_farfield,
                    ],
                    refinements=[
                        SurfaceRefinement(
                            name="does not work",
                            entities=[my_farfield.farfield],
                            max_edge_length=1e-4,
                        )
                    ],
                ),
            )

    # Test use of GhostSurface in boundary conditions
    with pytest.raises(
        ValueError,
        match=re.escape("Can not find any valid entity of type ['Surface'] from the input."),
    ):
        with SI_unit_system:
            my_farfield = AutomatedFarfield(name="my_farfield")
            _ = SimulationParams(
                meshing=MeshingParams(
                    volume_zones=[
                        my_farfield,
                    ],
                    refinements=[
                        SurfaceRefinement(
                            name="does not work",
                            entities=[my_farfield.farfield],
                            max_edge_length=1e-4,
                        )
                    ],
                ),
                models=[Wall(name="wall", surface=my_farfield.farfield)],
            )

    with SI_unit_system:
        my_farfield = AutomatedFarfield(name="my_farfield")
        _ = SimulationParams(
            models=[
                SlipWall(name="slipwall", entities=my_farfield.farfield),
                SymmetryPlane(name="symm_plane", entities=my_farfield.symmetry_planes),
            ],
        )

    with SI_unit_system:
        my_farfield = AutomatedFarfield(name="my_farfield")
        _ = SimulationParams(
            models=[
                Freestream(name="fs", entities=my_farfield.farfield),
            ],
        )

    # Test use of GhostSurface in SurfaceOutput
    with SI_unit_system:
        my_farfield = AutomatedFarfield(name="my_farfield")
        _ = SimulationParams(
            outputs=[
                SurfaceOutput(entities=my_farfield.farfield, output_fields=["Cp"]),
                SurfaceIntegralOutput(
                    name="prb 110",
                    entities=[
                        my_farfield.symmetry_planes,
                        Surface(name="surface2"),
                    ],
                    output_fields=["Cpt_user_defined"],
                ),
            ],
            user_defined_fields=[UserDefinedField(name="Cpt_user_defined", expression="Cp-123")],
        )


def test_automated_farfield_import_export():

    my_farfield = AutomatedFarfield(name="my_farfield")
    model_as_dict = my_farfield.model_dump()
    assert "private_attribute_entity" not in model_as_dict.keys()

    model_as_dict = {"name": "my_farfield", "method": "auto"}
    my_farfield = AutomatedFarfield(**model_as_dict)

    model_as_dict = {"name": "my_farfield"}
    my_farfield = AutomatedFarfield(**model_as_dict)

    with pytest.raises(
        ValueError,
        match=re.escape("Unable to extract tag using discriminator 'type'"),
    ):
        MeshingParams(**{"volume_zones": [model_as_dict]})

    model_as_dict = {"name": "my_farfield", "type": "AutomatedFarfield"}
    meshing = MeshingParams(**{"volume_zones": [model_as_dict]})
    assert isinstance(meshing.volume_zones[0], AutomatedFarfield)


def test_symmetric_existence(surface_mesh):

    farfield = AutomatedFarfield()
    with SI_unit_system:
        params = SimulationParams(
            operating_condition=AerospaceCondition(velocity_magnitude=1000),
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=0.001,
                    boundary_layer_growth_rate=1.1,
                ),
                volume_zones=[farfield],
            ),
            models=[
                Wall(surfaces=surface_mesh["*"]),
                Freestream(surfaces=[farfield.farfield]),
            ],
        )

    # Valid Symmetric but did not use it
    errors = _run_validation(params, surface_mesh)
    assert len(errors) == 1
    assert (
        "The following boundaries do not have a boundary condition: symmetric." in errors[0]["msg"]
    )

    params.models.append(SymmetryPlane(surfaces=[farfield.symmetry_planes]))
    errors = _run_validation(params, surface_mesh)
    assert errors is None

    # Invalid Symmetric
    params.meshing.defaults.planar_face_tolerance = 1e-100
    errors = _run_validation(params, surface_mesh)
    assert len(errors) == 1
    assert (
        "`symmetric` boundary will not be generated: model spans: [-4.1e-05, 1.2e+03], tolerance = 1e-100 x 2.5e+03 = 2.5e-97."
        in errors[0]["msg"]
    )

    # Invalid Symmetric but did not use it
    params.models.pop()
    errors = _run_validation(params, surface_mesh)
    assert errors is None


def test_user_defined_farfield_symmetry_plane(surface_mesh):
    farfield = UserDefinedFarfield()

    with SI_unit_system:
        params = SimulationParams(
            operating_condition=AerospaceCondition(velocity_magnitude=1),
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=0.001,
                    boundary_layer_growth_rate=1.1,
                ),
                volume_zones=[farfield],
            ),
            models=[
                Wall(surfaces=surface_mesh["*"]),
                SymmetryPlane(surfaces=farfield.symmetry_plane),
            ],
        )
    errors = _run_validation(params, surface_mesh, use_beta_mesher=True, use_geometry_AI=False)
    assert errors[0]["loc"] == ("models", 1, "entities", "stored_entities")
    assert (
        errors[0]["msg"]
        == "Value error, Symmetry plane of user defined farfield will only be generated when both GAI and beta mesher are used."
    )
    params.meshing.defaults.geometry_accuracy = 1 * u.mm
    errors = _run_validation(params, surface_mesh, use_beta_mesher=True, use_geometry_AI=True)
    assert errors is None


def test_rotated_symmetric_existence():
    geometry = Geometry.from_local_storage(
        geometry_id="geo-e5c01a98-2180-449e-b255-d60162854a83",
        local_storage_path=os.path.join("data", "geometry"),
        meta_data=GeometryMeta(
            **local_metadata_builder(
                id="geo-e5c01a98-2180-449e-b255-d60162854a83",
                name="Test",
                cloud_path_prefix="/",
                status="processed",
            )
        ),
    )

    geometry.group_faces_by_tag("faceId")
    geometry.group_edges_by_tag("edgeId")
    geometry.group_bodies_by_tag("groupByFile")

    farfield = AutomatedFarfield()
    body_name = "geo-9cafe735-1190-4e3e-978e-407271e254ed_cube-holes.csm"

    def _test_and_show_errors(geometry):
        # * 1: Missing symmetric
        with SI_unit_system:
            params = SimulationParams(
                operating_condition=AerospaceCondition(velocity_magnitude=1000),
                meshing=MeshingParams(
                    defaults=MeshingDefaults(
                        boundary_layer_first_layer_thickness=0.001,
                        boundary_layer_growth_rate=1.1,
                        geometry_accuracy=1e-7,
                        surface_max_edge_length=1e-9,
                    ),
                    volume_zones=[farfield],
                ),
                models=[
                    Wall(
                        surfaces=[
                            item for item in geometry["*"] if not item.name.endswith("face00005")
                        ]
                    ),
                    Freestream(surfaces=[farfield.farfield]),
                ],
            )

            processed_params = set_up_params_for_uploading(geometry, 1 * u.m, params, True, True)

        _, errors_1, _ = services.validate_model(
            params_as_dict=processed_params.model_dump(mode="json", exclude_none=True),
            validated_by=services.ValidationCalledBy.LOCAL,
            root_item_type="Geometry",
            validation_level="All",
        )

        # * 2: Missing boundary
        with SI_unit_system:
            params = SimulationParams(
                operating_condition=AerospaceCondition(velocity_magnitude=1000),
                meshing=MeshingParams(
                    defaults=MeshingDefaults(
                        boundary_layer_first_layer_thickness=0.001,
                        boundary_layer_growth_rate=1.1,
                        geometry_accuracy=1e-7,
                        surface_max_edge_length=1e-9,
                    ),
                    volume_zones=[farfield],
                ),
                models=[
                    Freestream(surfaces=[farfield.farfield]),
                    Wall(
                        surfaces=[
                            item
                            for item in geometry["*"]
                            if not item.name.endswith("face00005")
                            and item.name != "body00001_face00001"
                        ]
                    ),
                    SlipWall(surfaces=[farfield.symmetry_planes]),
                ],
            )

        processed_params = set_up_params_for_uploading(geometry, 1 * u.m, params, True, True)

        _, errors_2, _ = services.validate_model(
            params_as_dict=processed_params.model_dump(mode="json", exclude_none=True),
            validated_by=services.ValidationCalledBy.LOCAL,
            root_item_type="Geometry",
            validation_level="All",
        )
        print("# * 3: Deleted boundary")
        # * 3: Deleted boundary
        with SI_unit_system:
            params = SimulationParams(
                operating_condition=AerospaceCondition(velocity_magnitude=1000),
                meshing=MeshingParams(
                    defaults=MeshingDefaults(
                        boundary_layer_first_layer_thickness=0.001,
                        boundary_layer_growth_rate=1.1,
                        geometry_accuracy=1e-7,
                        surface_max_edge_length=1e-9,
                    ),
                    volume_zones=[farfield],
                ),
                models=[
                    Freestream(surfaces=[farfield.farfield]),
                    Wall(
                        surfaces=[
                            item for item in geometry["*"] if item.name != "body00001_face00005"
                        ]
                    ),
                    SlipWall(surfaces=[farfield.symmetry_planes]),
                ],
            )

        processed_params = set_up_params_for_uploading(geometry, 1 * u.m, params, True, True)

        _, errors_3, _ = services.validate_model(
            params_as_dict=processed_params.model_dump(mode="json", exclude_none=True),
            validated_by=services.ValidationCalledBy.LOCAL,
            root_item_type="Geometry",
            validation_level="All",
        )

        return errors_1, errors_2, errors_3

    errors_1, errors_2, errors_3 = _test_and_show_errors(geometry)

    assert len(errors_1) == 1
    assert (
        "The following boundaries do not have a boundary condition: symmetric."
        in errors_1[0]["msg"]
    )

    assert len(errors_2) == 1
    assert (
        "The following boundaries do not have a boundary condition: body00001_face00001."
        in errors_2[0]["msg"]
    )

    assert len(errors_3) == 1
    assert (
        "Boundary `body00002_face00005` will likely be deleted after mesh generation."
        in errors_3[0]["msg"]
    )

    geometry[body_name].transformation.angle_of_rotation = 90 * u.deg

    errors_1, errors_2, errors_3 = _test_and_show_errors(geometry)

    assert errors_1 is None
    assert errors_2 is None
    assert errors_3 is None

    geometry[body_name].transformation.angle_of_rotation = 0 * u.deg
    geometry[body_name].transformation.translation = [0, 0, 1e-9] * u.m

    errors_1, errors_2, errors_3 = _test_and_show_errors(geometry)

    assert errors_1 is None
    assert errors_2 is None
    assert errors_3 is None

    geometry[body_name].transformation.angle_of_rotation = 0 * u.deg
    geometry[body_name].transformation.translation = [0, 0, 1e-9] * u.m

    errors_1, errors_2, errors_3 = _test_and_show_errors(geometry)

    assert errors_1 is None
    assert errors_2 is None
    assert errors_3 is None

    geometry[body_name].transformation.translation = [0, 0, 0] * u.m
    geometry[body_name].transformation.scale = [0.5, 0.5, 1e-9]

    errors_1, errors_2, errors_3 = _test_and_show_errors(geometry)

    assert errors_1 is None
    assert errors_2 is None
    assert errors_3 is None
