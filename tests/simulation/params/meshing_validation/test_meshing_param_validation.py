from flow360 import u
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.meshing_param import snappy
from flow360.component.simulation.meshing_param.face_params import GeometryRefinement
from flow360.component.simulation.meshing_param.meshing_specs import (
    MeshingDefaults,
    VolumeMeshingDefaults,
)
from flow360.component.simulation.meshing_param.params import (
    MeshingParams,
    ModularMeshingWorkflow,
    VolumeMeshingParams,
)
from flow360.component.simulation.meshing_param.volume_params import (
    AutomatedFarfield,
    CustomZones,
    UniformRefinement,
    UserDefinedFarfield,
)
from flow360.component.simulation.primitives import Box, CustomVolume, Surface
from flow360.component.simulation.services import ValidationCalledBy, validate_model
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import SI_unit_system


def test_uniform_project_only_with_snappy():
    refinement = UniformRefinement(
        entities=[Box(center=(0, 0, 0) * u.m, size=(1, 1, 1) * u.m, name="box")],
        spacing=0.1 * u.m,
        project_to_surface=True,
    )
    with SI_unit_system:
        params_snappy = SimulationParams(
            meshing=ModularMeshingWorkflow(
                surface_meshing=snappy.SurfaceMeshingParams(
                    defaults=snappy.SurfaceMeshingDefaults(
                        min_spacing=1 * u.mm,
                        max_spacing=2 * u.mm,
                        gap_resolution=1 * u.mm,
                    )
                ),
                volume_meshing=VolumeMeshingParams(
                    defaults=VolumeMeshingDefaults(boundary_layer_first_layer_thickness=1 * u.mm),
                    refinements=[refinement],
                ),
                zones=[AutomatedFarfield()],
            )
        )

    _, errors, _ = validate_model(
        params_as_dict=params_snappy.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level="VolumeMesh",
    )
    assert errors is None

    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                volume_zones=[AutomatedFarfield()],
                refinements=[refinement],
                defaults=MeshingDefaults(
                    curvature_resolution_angle=12 * u.deg,
                    boundary_layer_growth_rate=1.1,
                    boundary_layer_first_layer_thickness=1e-5 * u.m,
                ),
            )
        )

    _, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="SurfaceMesh",
        validation_level="VolumeMesh",
    )

    assert len(errors) == 1
    assert errors[0]["msg"] == (
        "Value error, project_to_surface is supported only for snappyHexMesh."
    )
    assert errors[0]["loc"] == ("meshing", "refinements", 0, "UniformRefinement")


def test_per_face_min_passage_size_warning_without_remove_hidden_geometry():
    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    geometry_accuracy=0.01 * u.m,
                    surface_max_edge_length=0.1 * u.m,
                    remove_hidden_geometry=False,
                ),
                refinements=[
                    GeometryRefinement(
                        geometry_accuracy=0.01 * u.m,
                        min_passage_size=0.05 * u.m,
                        faces=[Surface(name="face1")],
                    ),
                ],
            ),
            private_attribute_asset_cache=AssetCache(
                use_geometry_AI=True,
                use_inhouse_mesher=True,
                project_length_unit=1 * u.m,
            ),
        )
    _, errors, warnings = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level="SurfaceMesh",
    )
    assert errors is None
    assert len(warnings) == 1
    assert "min_passage_size" in warnings[0]["msg"]
    assert "remove_hidden_geometry" in warnings[0]["msg"]

    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    geometry_accuracy=0.01 * u.m,
                    surface_max_edge_length=0.1 * u.m,
                    remove_hidden_geometry=True,
                ),
                refinements=[
                    GeometryRefinement(
                        geometry_accuracy=0.01 * u.m,
                        min_passage_size=0.05 * u.m,
                        faces=[Surface(name="face1")],
                    ),
                ],
            ),
            private_attribute_asset_cache=AssetCache(
                use_geometry_AI=True,
                use_inhouse_mesher=True,
                project_length_unit=1 * u.m,
            ),
        )
    _, errors, warnings = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level="SurfaceMesh",
    )
    assert errors is None
    assert warnings == []

    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    geometry_accuracy=0.01 * u.m,
                    surface_max_edge_length=0.1 * u.m,
                    remove_hidden_geometry=False,
                ),
                refinements=[
                    GeometryRefinement(
                        geometry_accuracy=0.01 * u.m,
                        faces=[Surface(name="face1")],
                    ),
                ],
            ),
            private_attribute_asset_cache=AssetCache(
                use_geometry_AI=True,
                use_inhouse_mesher=True,
                project_length_unit=1 * u.m,
            ),
        )
    _, errors, warnings = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level="SurfaceMesh",
    )
    assert errors is None
    assert warnings == []


def test_multi_zone_remove_hidden_geometry_warning():
    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    geometry_accuracy=0.01 * u.m,
                    surface_max_edge_length=0.1 * u.m,
                    remove_hidden_geometry=True,
                ),
                volume_zones=[
                    AutomatedFarfield(enclosed_entities=[Surface(name="face1")]),
                    CustomZones(
                        name="custom_zones",
                        entities=[
                            CustomVolume(
                                name="zone1",
                                bounding_entities=[Surface(name="face1"), Surface(name="face2")],
                            )
                        ],
                    ),
                ],
            ),
            private_attribute_asset_cache=AssetCache(
                use_geometry_AI=True,
                use_inhouse_mesher=True,
                project_length_unit=1 * u.m,
            ),
        )
    _, errors, warnings = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level="SurfaceMesh",
    )
    assert errors is None
    assert len(warnings) == 1
    assert (
        "removal of hidden geometry for multi-zone cases is not fully supported"
        in warnings[0]["msg"].lower()
    )

    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    geometry_accuracy=0.01 * u.m,
                    surface_max_edge_length=0.1 * u.m,
                    remove_hidden_geometry=True,
                ),
                volume_zones=[AutomatedFarfield()],
            ),
            private_attribute_asset_cache=AssetCache(
                use_geometry_AI=True,
                use_inhouse_mesher=True,
                project_length_unit=1 * u.m,
            ),
        )
    _, errors, warnings = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level="SurfaceMesh",
    )
    assert errors is None
    assert warnings == []

    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    geometry_accuracy=0.01 * u.m,
                    surface_max_edge_length=0.1 * u.m,
                    remove_hidden_geometry=False,
                ),
                volume_zones=[
                    AutomatedFarfield(enclosed_entities=[Surface(name="face1")]),
                    CustomZones(
                        name="custom_zones",
                        entities=[
                            CustomVolume(
                                name="zone1",
                                bounding_entities=[Surface(name="face1"), Surface(name="face2")],
                            )
                        ],
                    ),
                ],
            ),
            private_attribute_asset_cache=AssetCache(
                use_geometry_AI=True,
                use_inhouse_mesher=True,
                project_length_unit=1 * u.m,
            ),
        )
    _, errors, warnings = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level="SurfaceMesh",
    )
    assert errors is None
    assert warnings == []

    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    geometry_accuracy=0.01 * u.m,
                    surface_max_edge_length=0.1 * u.m,
                    remove_hidden_geometry=True,
                ),
                volume_zones=[
                    UserDefinedFarfield(),
                    CustomZones(
                        name="custom_zones",
                        entities=[
                            CustomVolume(
                                name="zone1",
                                bounding_entities=[Surface(name="face1"), Surface(name="face2")],
                            ),
                            CustomVolume(
                                name="zone2",
                                bounding_entities=[Surface(name="face3"), Surface(name="face4")],
                            ),
                        ],
                    ),
                ],
            ),
            private_attribute_asset_cache=AssetCache(
                use_geometry_AI=True,
                use_inhouse_mesher=True,
                project_length_unit=1 * u.m,
            ),
        )
    _, errors, warnings = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level="SurfaceMesh",
    )
    assert errors is None
    assert len(warnings) == 1
    assert (
        "removal of hidden geometry for multi-zone cases is not fully supported"
        in warnings[0]["msg"].lower()
    )

    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    geometry_accuracy=0.01 * u.m,
                    surface_max_edge_length=0.1 * u.m,
                    remove_hidden_geometry=True,
                ),
                volume_zones=[
                    UserDefinedFarfield(),
                    CustomZones(
                        name="custom_zones",
                        entities=[
                            CustomVolume(
                                name="zone1",
                                bounding_entities=[Surface(name="face1"), Surface(name="face2")],
                            )
                        ],
                    ),
                ],
            ),
            private_attribute_asset_cache=AssetCache(
                use_geometry_AI=True,
                use_inhouse_mesher=True,
                project_length_unit=1 * u.m,
            ),
        )
    _, errors, warnings = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level="SurfaceMesh",
    )
    assert errors is None
    assert warnings == []

    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    geometry_accuracy=0.01 * u.m,
                    surface_max_edge_length=0.1 * u.m,
                    remove_hidden_geometry=True,
                ),
                volume_zones=[
                    UserDefinedFarfield(),
                    CustomZones(
                        name="custom_zones",
                        entities=[
                            CustomVolume(
                                name="zone1",
                                bounding_entities=[Surface(name="face1"), Surface(name="face2")],
                            ),
                            CustomVolume(
                                name="zone2",
                                bounding_entities=[Surface(name="face3"), Surface(name="face4")],
                            ),
                        ],
                    ),
                ],
            ),
            private_attribute_asset_cache=AssetCache(
                use_geometry_AI=True,
                use_inhouse_mesher=True,
                project_length_unit=1 * u.m,
            ),
        )
    _, errors, warnings = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level="SurfaceMesh",
    )
    assert errors is None
    assert len(warnings) == 1
    assert (
        "removal of hidden geometry for multi-zone cases is not fully supported"
        in warnings[0]["msg"].lower()
    )

    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    geometry_accuracy=0.01 * u.m,
                    surface_max_edge_length=0.1 * u.m,
                    remove_hidden_geometry=True,
                ),
                volume_zones=[
                    UserDefinedFarfield(),
                    CustomZones(
                        name="custom_zones",
                        entities=[
                            CustomVolume(
                                name="zone1",
                                bounding_entities=[Surface(name="face1"), Surface(name="face2")],
                            )
                        ],
                    ),
                ],
            ),
            private_attribute_asset_cache=AssetCache(
                use_geometry_AI=True,
                use_inhouse_mesher=True,
                project_length_unit=1 * u.m,
            ),
        )
    _, errors, warnings = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level="SurfaceMesh",
    )
    assert errors is None
    assert warnings == []
