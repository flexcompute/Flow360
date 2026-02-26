import os
import re

import pytest
import unyt as u

from flow360.component.geometry import Geometry, GeometryMeta
from flow360.component.project import create_draft
from flow360.component.project_utils import set_up_params_for_uploading
from flow360.component.resource_base import local_metadata_builder
from flow360.component.simulation import services
from flow360.component.simulation.entity_info import SurfaceMeshEntityInfo
from flow360.component.simulation.entity_operation import CoordinateSystem
from flow360.component.simulation.framework.param_utils import AssetCache
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
from flow360.component.simulation.primitives import GhostSurface, Surface
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

    _, errors, warnings = services.validate_model(
        params_as_dict=params.model_dump(exclude_none=True),
        validated_by=services.ValidationCalledBy.LOCAL,
        root_item_type="SurfaceMesh",
        validation_level="All",
    )
    return errors, warnings


def test_automated_farfield_surface_usage():
    # Test use of GhostSurface in meshing via ValidationContext (Surface mesh + automated farfield):
    import pydantic as pd

    from flow360.component.simulation.validation.validation_context import (
        VOLUME_MESH,
        ParamsValidationInfo,
        ValidationContext,
    )

    with SI_unit_system:
        my_farfield = AutomatedFarfield(name="my_farfield")
        param_dict = {
            "meshing": {
                "type_name": "MeshingParams",
                "volume_zones": [
                    {"type": "AutomatedFarfield", "method": "auto"},
                ],
            },
            "private_attribute_asset_cache": {
                "use_inhouse_mesher": True,
                "use_geometry_AI": True,
                "project_entity_info": {"type_name": "SurfaceMeshEntityInfo"},
            },
        }
        info = ParamsValidationInfo(param_as_dict=param_dict, referenced_expressions=[])
        with ValidationContext(levels=VOLUME_MESH, info=info):
            with pytest.raises(pd.ValidationError):
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

    # Boundary condition (Wall) does not accept GhostSurface by type; keep original type-level error
    with SI_unit_system:
        my_farfield = AutomatedFarfield(name="my_farfield")
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Can not find any valid entity of type ['Surface', 'MirroredSurface', 'WindTunnelGhostSurface'] from the input."
            ),
        ):
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
                models=[Wall(name="wall", surfaces=[my_farfield.farfield])],
            )

    with SI_unit_system:
        my_farfield = AutomatedFarfield(name="my_farfield")
        _ = SimulationParams(
            models=[
                SlipWall(name="slipwall", entities=my_farfield.farfield),
                SymmetryPlane(name="symm_plane", entities=my_farfield.symmetry_plane),
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
                        my_farfield.symmetry_plane,
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
                Wall(surfaces=[s for s in surface_mesh["*"] if s.name != "preexistingSymmetry"]),
                Freestream(surfaces=[farfield.farfield]),
            ],
        )

    # Valid Symmetric but did not use it
    errors, _ = _run_validation(params, surface_mesh)
    assert len(errors) == 1
    assert (
        "The following boundaries do not have a boundary condition: symmetric." in errors[0]["msg"]
    )

    params.models.append(SymmetryPlane(surfaces=[farfield.symmetry_plane]))
    errors, warnings = _run_validation(params, surface_mesh)
    assert errors is None
    assert warnings == []

    # Invalid Symmetric
    params.meshing.defaults.planar_face_tolerance = 1e-100
    errors, _ = _run_validation(params, surface_mesh)
    assert len(errors) == 1
    assert (
        "`symmetric` boundary will not be generated: model spans: [-4.1e-05, 1.2e+03], tolerance = 1e-100 x 2.5e+03 = 2.5e-97."
        in errors[0]["msg"]
    )

    # Invalid Symmetric but did not use it
    params.models.pop()
    errors, warnings = _run_validation(params, surface_mesh)
    assert errors is None
    assert warnings == []


def test_user_defined_farfield_symmetry_plane(surface_mesh):
    farfield = UserDefinedFarfield(domain_type="half_body_positive_y")

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
                Wall(surfaces=[s for s in surface_mesh["*"] if s.name != "preexistingSymmetry"]),
                SymmetryPlane(surfaces=farfield.symmetry_plane),
            ],
        )
    errors, _ = _run_validation(params, surface_mesh, use_beta_mesher=True, use_geometry_AI=False)
    assert errors[0]["loc"][0] == "meshing"
    assert errors[0]["loc"][-1] == "domain_type"
    assert (
        errors[0]["msg"]
        == "Value error, `domain_type` is only supported when using both GAI surface mesher and beta volume mesher."
    )
    params.meshing.defaults.geometry_accuracy = 0.01 * u.m
    errors, warnings = _run_validation(
        params, surface_mesh, use_beta_mesher=True, use_geometry_AI=True
    )
    assert errors is None
    assert warnings == []


def test_user_defined_farfield_symmetry_plane_requires_half_domain(surface_mesh):
    farfield = UserDefinedFarfield(domain_type="full_body")

    with SI_unit_system:
        params = SimulationParams(
            operating_condition=AerospaceCondition(velocity_magnitude=1),
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=0.001,
                    boundary_layer_growth_rate=1.1,
                    geometry_accuracy=0.01 * u.m,
                ),
                volume_zones=[farfield],
            ),
            models=[
                Wall(surfaces=[s for s in surface_mesh["*"] if s.name != "preexistingSymmetry"]),
                SymmetryPlane(
                    surfaces=GhostSurface(name="symmetric", private_attribute_id="symmetric")
                ),
            ],
        )
    errors, _ = _run_validation(params, surface_mesh, use_beta_mesher=True, use_geometry_AI=True)
    assert errors[0]["loc"] == ("models", 1, "entities")
    assert (
        errors[0]["msg"]
        == "Value error, Symmetry plane of user defined farfield is only supported for half body domains."
    )


def test_user_defined_farfield_auto_symmetry_plane(surface_mesh):
    farfield = UserDefinedFarfield()

    with SI_unit_system:
        params = SimulationParams(
            operating_condition=AerospaceCondition(velocity_magnitude=1),
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=0.001,
                    boundary_layer_growth_rate=1.1,
                    geometry_accuracy=0.01 * u.m,
                ),
                volume_zones=[farfield],
            ),
            models=[
                Wall(surfaces=[s for s in surface_mesh["*"] if s.name != "preexistingSymmetry"]),
                SymmetryPlane(
                    surfaces=farfield.symmetry_plane,
                ),
            ],
        )
    errors, warnings = _run_validation(
        params, surface_mesh, use_beta_mesher=True, use_geometry_AI=True
    )
    assert errors is None
    assert warnings == []


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
                        geometry_accuracy=1e-4,
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

        _, errors_1, warnings_1 = services.validate_model(
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
                        geometry_accuracy=1e-4,
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
                    SlipWall(surfaces=[farfield.symmetry_plane]),
                ],
            )

        processed_params = set_up_params_for_uploading(geometry, 1 * u.m, params, True, True)

        _, errors_2, warnings_2 = services.validate_model(
            params_as_dict=processed_params.model_dump(mode="json", exclude_none=True),
            validated_by=services.ValidationCalledBy.LOCAL,
            root_item_type="Geometry",
            validation_level="All",
        )

        # * 3: Deleted boundary
        with SI_unit_system:
            params = SimulationParams(
                operating_condition=AerospaceCondition(velocity_magnitude=1000),
                meshing=MeshingParams(
                    defaults=MeshingDefaults(
                        boundary_layer_first_layer_thickness=0.001,
                        boundary_layer_growth_rate=1.1,
                        geometry_accuracy=1e-4,
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
                    SlipWall(surfaces=[farfield.symmetry_plane]),
                ],
            )

        processed_params = set_up_params_for_uploading(geometry, 1 * u.m, params, True, True)

        _, errors_3, _ = services.validate_model(
            params_as_dict=processed_params.model_dump(mode="json", exclude_none=True),
            validated_by=services.ValidationCalledBy.LOCAL,
            root_item_type="Geometry",
            validation_level="All",
        )

        return errors_1, errors_2, errors_3, warnings_1, warnings_2

    errors_1, errors_2, errors_3, warnings_1, warnings_2 = _test_and_show_errors(geometry)

    # With GAI enabled, missing BCs produce warnings instead of errors
    assert errors_1 is None or len(errors_1) == 0
    assert len(warnings_1) >= 1
    assert any(
        "The following boundaries do not have a boundary condition: symmetric." in w.get("msg", "")
        for w in warnings_1
    )

    assert errors_2 is None or len(errors_2) == 0
    assert len(warnings_2) >= 1
    assert any(
        "The following boundaries do not have a boundary condition: body00001_face00001."
        in w.get("msg", "")
        for w in warnings_2
    )

    assert len(errors_3) == 1
    assert (
        "Boundary `body00002_face00005` will likely be deleted after mesh generation."
        in errors_3[0]["msg"]
    )

    with create_draft(new_run_from=geometry) as draft:
        cs = CoordinateSystem(
            name="rotated",
            axis_of_rotation=(1.0, 0.0, 0.0),
            angle_of_rotation=90 * u.deg,
        )
        draft.coordinate_systems.assign(entities=draft.body_groups[body_name], coordinate_system=cs)
        errors_1, errors_2, errors_3, warnings_1, warnings_2 = _test_and_show_errors(geometry)

    assert errors_1 is None
    assert errors_2 is None
    assert errors_3 is None

    with create_draft(new_run_from=geometry) as draft:
        cs = CoordinateSystem(
            name="translated_small",
            translation=[0, 0, 1e-9] * u.m,
        )
        draft.coordinate_systems.assign(entities=draft.body_groups[body_name], coordinate_system=cs)
        errors_1, errors_2, errors_3, warnings_1, warnings_2 = _test_and_show_errors(geometry)

    assert errors_1 is None
    assert errors_2 is None
    assert errors_3 is None

    with create_draft(new_run_from=geometry) as draft:
        cs = CoordinateSystem(
            name="translated_small_repeat",
            translation=[0, 0, 1e-9] * u.m,
        )
        draft.coordinate_systems.assign(entities=draft.body_groups[body_name], coordinate_system=cs)
        errors_1, errors_2, errors_3, warnings_1, warnings_2 = _test_and_show_errors(geometry)

    assert errors_1 is None
    assert errors_2 is None
    assert errors_3 is None

    with create_draft(new_run_from=geometry) as draft:
        cs = CoordinateSystem(
            name="scaled",
            scale=(0.5, 0.5, 1e-9),
        )
        draft.coordinate_systems.assign(entities=draft.body_groups[body_name], coordinate_system=cs)
        errors_1, errors_2, errors_3, warnings_1, warnings_2 = _test_and_show_errors(geometry)

    assert errors_1 is None
    assert errors_2 is None
    assert errors_3 is None


def test_domain_type_bounding_box_check():
    # Case 1: Model does not cross Y=0 (Positive Half)
    # y range [1, 10]
    # Request half_body_positive_y -> Should pass (aligned)

    dummy_boundary = Surface(name="dummy", private_attribute_id="test-dummy-surface-id")

    asset_cache_positive = AssetCache(
        project_length_unit="m",
        use_inhouse_mesher=True,
        use_geometry_AI=True,
        project_entity_info=SurfaceMeshEntityInfo(
            global_bounding_box=[[0, 1, 0], [10, 10, 10]],
            ghost_entities=[],
            boundaries=[dummy_boundary],
        ),
    )

    farfield_pos = UserDefinedFarfield(domain_type="half_body_positive_y")

    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    planar_face_tolerance=0.01,
                    geometry_accuracy=1e-4,
                    boundary_layer_first_layer_thickness=1e-3,
                ),
                volume_zones=[farfield_pos],
            ),
            models=[Wall(entities=[dummy_boundary])],  # Assign BC to avoid missing BC error
            private_attribute_asset_cache=asset_cache_positive,
        )

    params_dict = params.model_dump(mode="json", exclude_none=True)
    _, errors, _ = services.validate_model(
        params_as_dict=params_dict,
        validated_by=services.ValidationCalledBy.LOCAL,
        root_item_type="SurfaceMesh",
        validation_level="All",
    )

    domain_errors = [
        e for e in (errors or []) if "The model does not cross the symmetry plane" in e["msg"]
    ]
    assert len(domain_errors) == 0

    # Case 2: Misaligned
    # Request half_body_negative_y on Positive Model -> Should Fail
    farfield_neg = UserDefinedFarfield(domain_type="half_body_negative_y")

    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    planar_face_tolerance=0.01,
                    geometry_accuracy=1e-4,
                    boundary_layer_first_layer_thickness=1e-3,
                ),
                volume_zones=[farfield_neg],
            ),
            models=[Wall(entities=[dummy_boundary])],
            private_attribute_asset_cache=asset_cache_positive,
        )

    params_dict = params.model_dump(mode="json", exclude_none=True)
    _, errors, _ = services.validate_model(
        params_as_dict=params_dict,
        validated_by=services.ValidationCalledBy.LOCAL,
        root_item_type="SurfaceMesh",
        validation_level="All",
    )

    assert errors is not None
    domain_errors = [e for e in errors if "The model does not cross the symmetry plane" in e["msg"]]
    assert len(domain_errors) == 1


def test_legacy_asset_missing_private_attributes():
    """Test that missing BCs are downgraded to warnings for legacy assets without private_attributes."""
    # Create surfaces without private_attributes to simulate legacy cloud assets
    wall_surface = Surface(name="wall", private_attribute_id="wall-1")
    farfield_surface = Surface(name="farfield_boundary", private_attribute_id="farfield-1")
    missing_surface = Surface(name="missing_bc", private_attribute_id="missing-1")

    # Explicitly set to None to simulate legacy assets
    wall_surface.private_attributes = None
    farfield_surface.private_attributes = None
    missing_surface.private_attributes = None

    asset_cache = AssetCache(
        project_length_unit="m",
        use_inhouse_mesher=True,
        use_geometry_AI=False,
        project_entity_info=SurfaceMeshEntityInfo(
            boundaries=[wall_surface, farfield_surface, missing_surface],
        ),
    )

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
                Wall(surfaces=[wall_surface]),  # Only assign BC to wall
                Freestream(
                    surfaces=[farfield_surface]
                ),  # Assign to farfield_boundary, not missing_bc
            ],
            private_attribute_asset_cache=asset_cache,
        )

    # Validate directly without set_up_params_for_uploading to preserve our None values
    _, errors, warnings = services.validate_model(
        params_as_dict=params.model_dump(mode="json", exclude_none=True),
        validated_by=services.ValidationCalledBy.LOCAL,
        root_item_type="SurfaceMesh",
        validation_level="All",
    )

    # Should get warnings, not errors (missing_bc has no BC assigned)
    assert errors is None, f"Expected no errors but got: {errors}"
    assert warnings is not None
    assert any(
        "missing_bc" in w.get("msg", "") for w in warnings
    ), f"Expected warning about missing_bc, got: {warnings}"
    assert any(
        "If these boundaries are valid" in w.get("msg", "") for w in warnings
    ), f"Expected specific warning message, got: {warnings}"
