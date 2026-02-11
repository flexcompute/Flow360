import json
import os

import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.framework.updater_utils import compare_values
from flow360.component.simulation.meshing_param import snappy
from flow360.component.simulation.meshing_param.face_params import (
    BoundaryLayer,
    PassiveSpacing,
    SurfaceRefinement,
)
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
    AxisymmetricRefinement,
    CustomZones,
    MeshSliceOutput,
    RotationVolume,
    StructuredBoxRefinement,
    UniformRefinement,
    UserDefinedFarfield,
    WheelBelts,
    WindTunnelFarfield,
)
from flow360.component.simulation.outputs.outputs import Slice
from flow360.component.simulation.primitives import (
    AxisymmetricBody,
    Box,
    CustomVolume,
    Cylinder,
    GhostCircularPlane,
    SeedpointVolume,
    Surface,
)
from flow360.component.simulation.services import ValidationCalledBy, validate_model
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.translator.volume_meshing_translator import (
    get_volume_meshing_json,
)
from flow360.component.simulation.unit_system import LengthType, SI_unit_system
from flow360.component.simulation.utils import model_attribute_unlock
from flow360.component.simulation.validation.validation_context import VOLUME_MESH
from tests.simulation.conftest import AssetBase


class TempSurfaceMesh(AssetBase):
    """Mimicing the final SurfaceMesh class"""

    fname: str
    mesh_unit: LengthType.Positive

    def _get_meta_data(self):
        if self.fname == "om6wing.cgns":
            return {
                "surfaces": {
                    "wing": {},
                },
                "mesh_unit": {"units": "m", "value": 1.0},
            }
        else:
            raise ValueError("Invalid file name")

    def _populate_registry(self):
        self.mesh_unit = LengthType.validate(self._get_meta_data()["mesh_unit"])
        for surface_name in self._get_meta_data()["surfaces"]:
            self.internal_registry.register(Surface(name=surface_name))

    def __init__(self, file_name: str):
        super().__init__()
        self.fname = file_name
        self._populate_registry()


@pytest.fixture()
def get_surface_mesh():
    return TempSurfaceMesh("om6wing.cgns")


@pytest.fixture()
def get_test_param():
    def _make(beta_mesher: bool = True):
        with SI_unit_system:
            base_cylinder = Cylinder(
                name="cylinder_1",
                outer_radius=1.1,
                height=2 * u.m,
                axis=(0, 1, 0),
                center=(0.7, -1.0, 0),
            )
            rotor_disk_cylinder = Cylinder(
                name="enclosed",
                outer_radius=1.1,
                height=0.15 * u.m,
                axis=(0, 1, 0),
                center=(0.7, -1.0, 0),
            )
            inner_cylinder = Cylinder(
                name="inner",
                outer_radius=0.75,
                height=0.5,
                axis=(0, 0, 1),
                center=(0, 0, 0),
            )
            mid_cylinder = Cylinder(
                name="mid",
                outer_radius=2,
                height=2,
                axis=(0, 1, 0),
                center=(0, 0, 0),
            )
            cylinder_2 = Cylinder(
                name="2",
                outer_radius=2,
                height=2,
                axis=(0, 1, 0),
                center=(0, 5, 0),
            )
            cylinder_3 = Cylinder(
                name="3",
                inner_radius=1.5,
                outer_radius=2,
                height=2,
                axis=(0, 1, 0),
                center=(0, -5, 0),
            )
            cylinder_outer = Cylinder(
                name="outer",
                inner_radius=0,
                outer_radius=8,
                height=6,
                axis=(1, 0, 0),
                center=(0, 0, 0),
            )
            cone_frustum = AxisymmetricBody(
                name="cone",
                axis=(1, 0, 1),
                center=(0, 0, 0),
                profile_curve=[(-1, 0), (-1, 1), (1, 2), (1, 0)],
            )
            cone_frustum_mm_curve = AxisymmetricBody(
                name="cone_mm_curve",
                axis=(1, 0, 1),
                center=(0, 1, 0) * u.cm,
                profile_curve=[(-1, 0) * u.mm, (-1, 1) * u.mm, (1, 2) * u.mm, (1, 0) * u.mm],
            )

            porous_medium = Box.from_principal_axes(
                name="porousRegion",
                center=(0, 1, 1),
                size=(1, 2, 1),
                axes=((2, 2, 0), (-2, 2, 0)),
            )

            # Build refinements
            refinements = [
                UniformRefinement(
                    entities=[
                        base_cylinder,
                        Box.from_principal_axes(
                            name="MyBox",
                            center=(0, 1, 2),
                            size=(4, 5, 6),
                            axes=((2, 2, 0), (-2, 2, 0)),
                        ),
                    ],
                    spacing=7.5 * u.cm,
                ),
                AxisymmetricRefinement(
                    entities=[rotor_disk_cylinder],
                    spacing_axial=20 * u.cm,
                    spacing_radial=0.2,
                    spacing_circumferential=20 * u.cm,
                ),
                PassiveSpacing(entities=[Surface(name="passive1")], type="projected"),
                PassiveSpacing(entities=[Surface(name="passive2")], type="unchanged"),
                BoundaryLayer(
                    entities=[Surface(name="boundary1")],
                    first_layer_thickness=0.5 * u.m,
                    growth_rate=(1.3 if beta_mesher else None),
                ),
            ]
            if beta_mesher:
                refinements.append(
                    StructuredBoxRefinement(
                        entities=[porous_medium],
                        spacing_axis1=7.5 * u.cm,
                        spacing_axis2=10 * u.cm,
                        spacing_normal=15 * u.cm,
                    )
                )

            # Build volume_zones
            volume_zones = []
            if beta_mesher:
                volume_zones.append(
                    CustomZones(
                        name="custom_zones",
                        entities=[
                            CustomVolume(
                                name="custom_volume-1",
                                boundaries=[Surface(name="interface1"), Surface(name="interface2")],
                            )
                        ],
                    )
                )
            volume_zones.append(
                UserDefinedFarfield(domain_type=("half_body_negative_y" if beta_mesher else None))
            )
            volume_zones.append(
                RotationVolume(
                    name="we_do_not_use_this_anyway",
                    entities=inner_cylinder,
                    spacing_axial=20 * u.cm,
                    spacing_radial=0.2,
                    spacing_circumferential=20 * u.cm,
                    enclosed_entities=[
                        Surface(name="hub"),
                        Surface(name="blade1"),
                        Surface(name="blade2"),
                        Surface(name="blade3"),
                        *([cone_frustum] if beta_mesher else []),
                    ],
                )
            )
            volume_zones.append(
                RotationVolume(
                    entities=mid_cylinder,
                    spacing_axial=20 * u.cm,
                    spacing_radial=0.2,
                    spacing_circumferential=20 * u.cm,
                    enclosed_entities=[inner_cylinder],
                )
            )
            volume_zones.append(
                RotationVolume(
                    entities=cylinder_2,
                    spacing_axial=20 * u.cm,
                    spacing_radial=0.2,
                    spacing_circumferential=20 * u.cm,
                    enclosed_entities=[
                        rotor_disk_cylinder,
                        *([porous_medium] if beta_mesher else []),
                    ],
                )
            )
            volume_zones.append(
                RotationVolume(
                    entities=cylinder_3,
                    spacing_axial=20 * u.cm,
                    spacing_radial=0.2,
                    spacing_circumferential=20 * u.cm,
                )
            )
            volume_zones.append(
                RotationVolume(
                    entities=cylinder_outer,
                    spacing_axial=40 * u.cm,
                    spacing_radial=0.4,
                    spacing_circumferential=40 * u.cm,
                    enclosed_entities=[mid_cylinder, rotor_disk_cylinder, cylinder_2, cylinder_3],
                )
            )
            if beta_mesher:
                volume_zones.append(
                    RotationVolume(
                        entities=cone_frustum,
                        spacing_axial=40 * u.cm,
                        spacing_radial=0.4,
                        spacing_circumferential=20 * u.cm,
                    )
                )
                volume_zones.append(
                    RotationVolume(
                        entities=cone_frustum_mm_curve,
                        spacing_axial=40 * u.mm,
                        spacing_radial=0.4,
                        spacing_circumferential=20 * u.mm,
                    )
                )

            # Build mesh slice outputs
            meshSliceOutputs = []
            meshSliceOutputs.append(
                MeshSliceOutput(
                    name="slice_output",
                    entities=[
                        Slice(
                            name=f"test_slice_y_normal",
                            origin=(0.1, 0.2, 0.3),
                            normal=(0, 1, 0),
                        ),
                        Slice(
                            name=f"test_slice_z_normal",
                            origin=(0.6, 0.1, 0.4),
                            normal=(0, 0, 1),
                        ),
                    ],
                )
            )

            meshSliceOutputs.append(
                MeshSliceOutput(
                    name="slice_output_2",
                    entities=[
                        Slice(
                            name=f"crinkled_slice_with_cutoff",
                            origin=(0.1, 0.2, 0.3),
                            normal=(0, 1, 1),
                        ),
                    ],
                    include_crinkled_slices=True,
                    cutoff_radius=10.0,
                )
            )

            meshSliceOutputs.append(
                MeshSliceOutput(
                    name="slice_output_3",
                    entities=[
                        Slice(
                            name=f"crinkled_slice_without_cutoff",
                            origin=(0.5, 0.6, 0.7),
                            normal=(-1, 0, 0),
                        ),
                    ],
                    include_crinkled_slices=True,
                )
            )

            param = SimulationParams(
                meshing=MeshingParams(
                    refinement_factor=1.45,
                    defaults=MeshingDefaults(
                        boundary_layer_first_layer_thickness=1.35e-06 * u.m,
                        boundary_layer_growth_rate=1 + 0.04,
                    ),
                    refinements=refinements,
                    volume_zones=volume_zones,
                    outputs=meshSliceOutputs,
                ),
                private_attribute_asset_cache=AssetCache(use_inhouse_mesher=beta_mesher),
            )
            return param

    return _make


@pytest.fixture()
def get_test_param_modular():
    with SI_unit_system:
        base_cylinder = Cylinder(
            name="cylinder_1",
            outer_radius=1.1,
            height=2 * u.m,
            axis=(0, 1, 0),
            center=(0.7, -1.0, 0),
        )
        rotor_disk_cylinder = Cylinder(
            name="enclosed",
            outer_radius=1.1,
            height=0.15 * u.m,
            axis=(0, 1, 0),
            center=(0.7, -1.0, 0),
        )
        inner_cylinder = Cylinder(
            name="inner",
            outer_radius=0.75,
            height=0.5,
            axis=(0, 0, 1),
            center=(0, 0, 0),
        )
        mid_cylinder = Cylinder(
            name="mid",
            outer_radius=2,
            height=2,
            axis=(0, 1, 0),
            center=(0, 0, 0),
        )
        cylinder_2 = Cylinder(
            name="2",
            outer_radius=2,
            height=2,
            axis=(0, 1, 0),
            center=(0, 5, 0),
        )
        cylinder_3 = Cylinder(
            name="3", inner_radius=1.5, outer_radius=2, height=2, axis=(0, 1, 0), center=(0, -5, 0)
        )
        cylinder_outer = Cylinder(
            name="outer",
            inner_radius=0,
            outer_radius=8,
            height=6,
            axis=(1, 0, 0),
            center=(0, 0, 0),
        )
        cone_frustum = AxisymmetricBody(
            name="cone",
            axis=(1, 0, 1),
            center=(0, 0, 0),
            profile_curve=[(-1, 0), (-1, 1), (1, 2), (1, 0)],
        )
        cone_frustum_mm_curve = AxisymmetricBody(
            name="cone_mm_curve",
            axis=(1, 0, 1),
            center=(0, 1, 0) * u.cm,
            profile_curve=[(-1, 0) * u.mm, (-1, 1) * u.mm, (1, 2) * u.mm, (1, 0) * u.mm],
        )

        porous_medium = Box.from_principal_axes(
            name="porousRegion",
            center=(0, 1, 1),
            size=(1, 2, 1),
            axes=((2, 2, 0), (-2, 2, 0)),
        )
        param = SimulationParams(
            meshing=ModularMeshingWorkflow(
                volume_meshing=VolumeMeshingParams(
                    defaults=VolumeMeshingDefaults(
                        boundary_layer_first_layer_thickness=1.35e-06 * u.m,
                        boundary_layer_growth_rate=1 + 0.04,
                    ),
                    refinement_factor=1.45,
                    refinements=[
                        UniformRefinement(
                            entities=[
                                base_cylinder,
                                Box.from_principal_axes(
                                    name="MyBox",
                                    center=(0, 1, 2),
                                    size=(4, 5, 6),
                                    axes=((2, 2, 0), (-2, 2, 0)),
                                ),
                            ],
                            spacing=7.5 * u.cm,
                        ),
                        AxisymmetricRefinement(
                            entities=[rotor_disk_cylinder],
                            spacing_axial=20 * u.cm,
                            spacing_radial=0.2,
                            spacing_circumferential=20 * u.cm,
                        ),
                        StructuredBoxRefinement(
                            entities=[porous_medium],
                            spacing_axis1=7.5 * u.cm,
                            spacing_axis2=10 * u.cm,
                            spacing_normal=15 * u.cm,
                        ),
                        PassiveSpacing(entities=[Surface(name="passive1")], type="projected"),
                        PassiveSpacing(entities=[Surface(name="passive2")], type="unchanged"),
                        BoundaryLayer(
                            entities=[Surface(name="boundary1")],
                            first_layer_thickness=0.5 * u.m,
                            growth_rate=1.3,
                        ),
                    ],
                ),
                zones=[
                    CustomZones(
                        name="custom_zones",
                        entities=[
                            CustomVolume(
                                name="custom_volume-1",
                                boundaries=[Surface(name="interface1"), Surface(name="interface2")],
                            ),
                        ],
                    ),
                    RotationVolume(
                        name="we_do_not_use_this_anyway",
                        entities=inner_cylinder,
                        spacing_axial=20 * u.cm,
                        spacing_radial=0.2,
                        spacing_circumferential=20 * u.cm,
                        enclosed_entities=[
                            Surface(name="hub"),
                            Surface(name="blade1"),
                            Surface(name="blade2"),
                            Surface(name="blade3"),
                            cone_frustum,
                        ],
                    ),
                    RotationVolume(
                        entities=mid_cylinder,
                        spacing_axial=20 * u.cm,
                        spacing_radial=0.2,
                        spacing_circumferential=20 * u.cm,
                        enclosed_entities=[inner_cylinder],
                    ),
                    RotationVolume(
                        entities=cylinder_2,
                        spacing_axial=20 * u.cm,
                        spacing_radial=0.2,
                        spacing_circumferential=20 * u.cm,
                        enclosed_entities=[rotor_disk_cylinder, porous_medium],
                    ),
                    RotationVolume(
                        entities=cylinder_3,
                        spacing_axial=20 * u.cm,
                        spacing_radial=0.2,
                        spacing_circumferential=20 * u.cm,
                    ),
                    RotationVolume(
                        entities=cylinder_outer,
                        spacing_axial=40 * u.cm,
                        spacing_radial=0.4,
                        spacing_circumferential=40 * u.cm,
                        enclosed_entities=[
                            mid_cylinder,
                            rotor_disk_cylinder,
                            cylinder_2,
                            cylinder_3,
                        ],
                    ),
                    RotationVolume(
                        entities=cone_frustum,
                        spacing_axial=40 * u.cm,
                        spacing_radial=0.4,
                        spacing_circumferential=20 * u.cm,
                    ),
                    RotationVolume(
                        entities=cone_frustum_mm_curve,
                        spacing_axial=40 * u.mm,
                        spacing_radial=0.4,
                        spacing_circumferential=20 * u.mm,
                    ),
                ],
            ),
            private_attribute_asset_cache=AssetCache(use_inhouse_mesher=True),
        )
    return param


@pytest.fixture()
def get_test_param_w_seedpoints():
    with SI_unit_system:
        param = SimulationParams(
            meshing=ModularMeshingWorkflow(
                surface_meshing=snappy.SurfaceMeshingParams(
                    defaults=snappy.SurfaceMeshingDefaults(
                        min_spacing=1, max_spacing=2, gap_resolution=1
                    )
                ),
                volume_meshing=VolumeMeshingParams(
                    defaults=VolumeMeshingDefaults(
                        boundary_layer_first_layer_thickness=1.35e-06 * u.m,
                        boundary_layer_growth_rate=1 + 0.04,
                    ),
                    refinement_factor=1.45,
                    refinements=[],
                ),
                zones=[
                    CustomZones(
                        entities=[
                            SeedpointVolume(name="fluid", point_in_mesh=(0, 0, 0)),
                            SeedpointVolume(name="radiator", point_in_mesh=(1, 1, 1)),
                        ]
                    )
                ],
            ),
            private_attribute_asset_cache=AssetCache(use_inhouse_mesher=True),
        )
    return param


def test_param_to_json(get_test_param, get_surface_mesh, get_test_param_modular):
    translated = get_volume_meshing_json(get_test_param(), get_surface_mesh.mesh_unit)
    translated_modular = get_volume_meshing_json(get_test_param_modular, get_surface_mesh.mesh_unit)
    ref_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ref",
        "volume_meshing",
        "ref_param_to_json_inhouse.json",
    )
    with open(ref_path, "r") as fh:
        ref_dict = json.load(fh)

    assert compare_values(translated, ref_dict)
    ref_dict["farfield"].pop("domainType", None)
    ref_dict.pop("meshSliceOutput", None)
    assert compare_values(translated_modular, ref_dict)


def test_user_defined_farfield(get_test_param, get_surface_mesh):
    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(boundary_layer_first_layer_thickness=100),
                volume_zones=[UserDefinedFarfield()],
            )
        )
        params_modular = SimulationParams(
            meshing=ModularMeshingWorkflow(
                surface_meshing=snappy.SurfaceMeshingParams(
                    defaults=snappy.SurfaceMeshingDefaults(
                        min_spacing=1, max_spacing=2, gap_resolution=1
                    )
                ),
                volume_meshing=VolumeMeshingParams(
                    defaults=VolumeMeshingDefaults(boundary_layer_first_layer_thickness=100),
                ),
                zones=[
                    CustomZones(
                        entities=[SeedpointVolume(point_in_mesh=[0, 0, 0], name="farfield")]
                    )
                ],
            )
        )
    translated = get_volume_meshing_json(params, get_surface_mesh.mesh_unit)
    translated_modular = get_volume_meshing_json(params_modular, get_surface_mesh.mesh_unit)
    reference_standard = {
        "refinementFactor": 1.0,
        "farfield": {"type": "user-defined"},
        "volume": {
            "firstLayerThickness": 100.0,
            "growthRate": 1.2,
            "gapTreatmentStrength": 0.0,
        },
        "faces": {},
    }
    reference_snappy_modular = {
        "refinementFactor": 1.0,
        "farfield": {"type": "user-defined"},
        "volume": {
            "firstLayerThickness": 100.0,
            "growthRate": 1.2,
            "gapTreatmentStrength": 0.0,
        },
        "faces": {},
        "zones": [{"name": "farfield", "pointInMesh": [0, 0, 0]}],
    }
    assert sorted(translated.items()) == sorted(reference_standard.items())
    assert sorted(translated_modular.items()) == sorted(reference_snappy_modular.items())


def test_seedpoint_zones(get_test_param_w_seedpoints, get_surface_mesh):
    translated_modular = get_volume_meshing_json(
        get_test_param_w_seedpoints, get_surface_mesh.mesh_unit
    )

    reference = {
        "refinementFactor": 1.45,
        "farfield": {"type": "user-defined"},
        "volume": {
            "firstLayerThickness": 1.35e-06,
            "growthRate": 1.04,
            "gapTreatmentStrength": 1.0,
            "planarFaceTolerance": 1e-6,
            "slidingInterfaceTolerance": 1e-2,
            "numBoundaryLayers": -1,
        },
        "faces": {},
        "zones": [
            {
                "name": "fluid",
                "pointInMesh": [0, 0, 0],
            },
            {
                "name": "radiator",
                "pointInMesh": [1, 1, 1],
            },
        ],
    }

    assert sorted(translated_modular.items()) == sorted(reference.items())


def test_param_to_json_legacy_mesher(get_test_param, get_surface_mesh):
    # Build params using legacy mesher (non-beta)
    params = get_test_param(beta_mesher=False)
    # Ensure no illegal features used:
    params, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="Geometry",
        validation_level=VOLUME_MESH,
    )
    assert errors is None
    translated = get_volume_meshing_json(params, get_surface_mesh.mesh_unit)
    ref_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ref",
        "volume_meshing",
        "ref_param_to_json_legacy.json",
    )
    with open(ref_path, "r") as fh:
        ref_dict = json.load(fh)
    assert compare_values(translated, ref_dict)


def test_custom_zones_tetrahedra(get_test_param, get_surface_mesh):
    """Base branch: No enforceTetrahedralElements emitted; ensure translator does not include it."""
    params = get_test_param()
    translated = get_volume_meshing_json(params, get_surface_mesh.mesh_unit)
    assert "zones" in translated and len(translated["zones"]) > 0
    assert all("enforceTetrahedralElements" not in z for z in translated["zones"])  # type: ignore


def test_custom_zones_element_type_tetrahedra(get_surface_mesh):
    """Test that element_type='tetrahedra' is correctly translated to enforceTetrahedralElements=True."""
    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                refinement_factor=1.0,
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=1e-6 * u.m,
                    boundary_layer_growth_rate=1.2,
                ),
                volume_zones=[
                    CustomZones(
                        name="custom_zones_tetrahedral",
                        entities=[
                            CustomVolume(
                                name="tetrahedral_zone",
                                boundaries=[Surface(name="boundary1"), Surface(name="boundary2")],
                            )
                        ],
                        element_type="tetrahedra",
                    ),
                    UserDefinedFarfield(),
                ],
            ),
            private_attribute_asset_cache=AssetCache(use_inhouse_mesher=True),
        )

    translated = get_volume_meshing_json(params, get_surface_mesh.mesh_unit)
    assert "zones" in translated
    assert len(translated["zones"]) == 1
    assert translated["zones"][0]["name"] == "tetrahedral_zone"
    assert "enforceTetrahedralElements" in translated["zones"][0]
    assert translated["zones"][0]["enforceTetrahedralElements"] is True


def test_custom_zones_element_type_mixed(get_surface_mesh):
    """Test that element_type='mixed' (default) does not include enforceTetrahedralElements."""
    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                refinement_factor=1.0,
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=1e-6 * u.m,
                    boundary_layer_growth_rate=1.2,
                ),
                volume_zones=[
                    CustomZones(
                        name="custom_zones_mixed",
                        entities=[
                            CustomVolume(
                                name="mixed_zone",
                                boundaries=[Surface(name="boundary1"), Surface(name="boundary2")],
                            )
                        ],
                        element_type="mixed",
                    ),
                    UserDefinedFarfield(),
                ],
            ),
            private_attribute_asset_cache=AssetCache(use_inhouse_mesher=True),
        )

    translated = get_volume_meshing_json(params, get_surface_mesh.mesh_unit)
    assert "zones" in translated
    assert len(translated["zones"]) == 1
    assert translated["zones"][0]["name"] == "mixed_zone"
    assert "enforceTetrahedralElements" not in translated["zones"][0]


def test_passive_spacing_with_ghost_symmetry_in_faces(get_surface_mesh):
    # PassiveSpacing using a GhostSurface (UserDefinedFarfield.symmetry_plane)
    with SI_unit_system:
        far = UserDefinedFarfield(domain_type="half_body_positive_y")
        params = SimulationParams(
            meshing=MeshingParams(
                refinement_factor=1.0,
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=1e-6 * u.m,
                    boundary_layer_growth_rate=1.2,
                ),
                volume_zones=[far],
                refinements=[
                    PassiveSpacing(entities=[far.symmetry_plane], type="projected"),
                ],
            ),
            private_attribute_asset_cache=AssetCache(
                use_inhouse_mesher=True,
                use_geometry_AI=True,
            ),
        )

    translated = get_volume_meshing_json(params, get_surface_mesh.mesh_unit)
    assert "faces" in translated
    assert "symmetric" in translated["faces"]
    assert translated["faces"]["symmetric"]["type"] == "projectAnisoSpacing"


@pytest.mark.parametrize(
    "use_gai,use_beta",
    [
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_user_defined_farfield_ghost_symmetry_requires_gai_and_beta(
    use_gai, use_beta, get_surface_mesh
):
    # Using GhostCircularPlane("symmetric") must require both GAI and beta mesher for user-defined farfield
    import pydantic as pd

    from flow360.component.simulation.validation.validation_context import (
        VOLUME_MESH,
        ParamsValidationInfo,
        ValidationContext,
    )

    with SI_unit_system:
        # Build minimal param dict for validation info
        param_dict = {
            "meshing": {
                "type_name": "MeshingParams",
                "volume_zones": [
                    {"type": "UserDefinedFarfield", "domain_type": "half_body_positive_y"}
                ],
            },
            "private_attribute_asset_cache": {
                "use_inhouse_mesher": use_beta,
                "use_geometry_AI": use_gai,
            },
        }
        info = ParamsValidationInfo(param_as_dict=param_dict, referenced_expressions=[])
        with ValidationContext(levels=VOLUME_MESH, info=info):
            with pytest.raises(
                pd.ValidationError,
                match="only be generated when both GAI and beta mesher are used",
            ):
                PassiveSpacing(entities=[GhostCircularPlane(name="symmetric")], type="projected")


def test_user_defined_farfield_ghost_symmetry_passes_with_gai_and_beta(get_surface_mesh):
    # Positive case: both flags enabled and half-body domain -> validator should pass
    from flow360.component.simulation.validation.validation_context import (
        VOLUME_MESH,
        ParamsValidationInfo,
        ValidationContext,
    )

    with SI_unit_system:
        param_dict = {
            "meshing": {
                "type_name": "MeshingParams",
                "volume_zones": [
                    {"type": "UserDefinedFarfield", "domain_type": "half_body_positive_y"}
                ],
            },
            "private_attribute_asset_cache": {
                "use_inhouse_mesher": True,
                "use_geometry_AI": True,
            },
        }
        info = ParamsValidationInfo(param_as_dict=param_dict, referenced_expressions=[])
        with ValidationContext(levels=VOLUME_MESH, info=info):
            PassiveSpacing(entities=[GhostCircularPlane(name="symmetric")], type="projected")


def test_geometry_auto_farfield_requires_beta_for_ghost_in_face_refinements():
    # Geometry + automated farfield: both PassiveSpacing and SurfaceRefinement require beta mesher
    import pydantic as pd

    from flow360.component.simulation.validation.validation_context import (
        VOLUME_MESH,
        ParamsValidationInfo,
        ValidationContext,
    )

    with SI_unit_system:
        # no beta -> should fail
        param_dict = {
            "meshing": {
                "type_name": "MeshingParams",
                "volume_zones": [{"type": "AutomatedFarfield", "method": "auto"}],
            },
            "private_attribute_asset_cache": {
                "use_inhouse_mesher": False,
                "use_geometry_AI": True,
                "project_entity_info": {"type_name": "GeometryEntityInfo"},
            },
        }
        info = ParamsValidationInfo(param_as_dict=param_dict, referenced_expressions=[])
        with ValidationContext(levels=VOLUME_MESH, info=info):
            with pytest.raises(pd.ValidationError, match="requires beta mesher"):
                SurfaceRefinement(
                    entities=[GhostCircularPlane(name="symmetric")], max_edge_length=0.1
                )
            with pytest.raises(pd.ValidationError, match="requires beta mesher"):
                PassiveSpacing(entities=[GhostCircularPlane(name="symmetric")], type="projected")

    with SI_unit_system:
        # beta -> should pass
        param_dict = {
            "meshing": {
                "type_name": "MeshingParams",
                "volume_zones": [{"type": "AutomatedFarfield", "method": "auto"}],
            },
            "private_attribute_asset_cache": {
                "use_inhouse_mesher": True,
                "use_geometry_AI": False,
                "project_entity_info": {"type_name": "GeometryEntityInfo"},
            },
        }
        info = ParamsValidationInfo(param_as_dict=param_dict, referenced_expressions=[])
        with ValidationContext(levels=VOLUME_MESH, info=info):
            SurfaceRefinement(entities=[GhostCircularPlane(name="symmetric")], max_edge_length=0.1)
            PassiveSpacing(entities=[GhostCircularPlane(name="symmetric")], type="projected")


def test_surface_mesh_auto_farfield_only_passive_spacing_allows_ghost():
    # Surface mesh + automated farfield: allow ghost for PassiveSpacing only; SR should fail
    import pydantic as pd

    from flow360.component.simulation.validation.validation_context import (
        VOLUME_MESH,
        ParamsValidationInfo,
        ValidationContext,
    )

    with SI_unit_system:
        param_dict = {
            "meshing": {
                "type_name": "MeshingParams",
                "volume_zones": [{"type": "AutomatedFarfield", "method": "auto"}],
            },
            "private_attribute_asset_cache": {
                "use_inhouse_mesher": True,
                "use_geometry_AI": True,
                "project_entity_info": {"type_name": "SurfaceMeshEntityInfo"},
            },
        }
        info = ParamsValidationInfo(param_as_dict=param_dict, referenced_expressions=[])
        with ValidationContext(levels=VOLUME_MESH, info=info):
            # SurfaceRefinement should reject ghost
            with pytest.raises(pd.ValidationError, match="not allowed for SurfaceRefinement"):
                SurfaceRefinement(
                    entities=[GhostCircularPlane(name="symmetric")], max_edge_length=0.1
                )
            # PassiveSpacing should accept ghost
            PassiveSpacing(entities=[GhostCircularPlane(name="symmetric")], type="projected")


def test_surface_mesh_user_defined_farfield_disallow_any_ghost():
    # Surface mesh + user-defined farfield: disallow ghost in both SR and PS
    import pydantic as pd

    from flow360.component.simulation.validation.validation_context import (
        VOLUME_MESH,
        ParamsValidationInfo,
        ValidationContext,
    )

    with SI_unit_system:
        param_dict = {
            "meshing": {
                "type_name": "MeshingParams",
                "volume_zones": [{"type": "UserDefinedFarfield"}],
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
                SurfaceRefinement(
                    entities=[GhostCircularPlane(name="symmetric")], max_edge_length=0.1
                )
            with pytest.raises(pd.ValidationError):
                PassiveSpacing(entities=[GhostCircularPlane(name="symmetric")], type="projected")


def test_farfield_relative_size():
    with SI_unit_system:
        param = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=1e-3, boundary_layer_growth_rate=1.25
                ),
                volume_zones=[AutomatedFarfield(method="quasi-3d", relative_size=100.0)],
            )
        )
    translated = get_volume_meshing_json(param, u.m)
    ref_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ref",
        "volume_meshing",
        "ref_param_to_json_legacy_farfield_relative_size.json",
    )
    with open(ref_path, "r") as fh:
        ref_dict = json.load(fh)
    assert compare_values(translated, ref_dict)


def test_analytic_wind_tunnel_farfield():
    with SI_unit_system:
        wind_tunnel = WindTunnelFarfield(
            width=10,
            height=10,
            inlet_x_position=-5,
            outlet_x_position=15,
            floor_z_position=0,
            floor_type=WheelBelts(
                central_belt_x_range=(-1, 6),
                central_belt_width=1.2,
                front_wheel_belt_x_range=(-0.3, 0.5),
                front_wheel_belt_y_range=(0.7, 1.2),
                rear_wheel_belt_x_range=(2.6, 3.8),
                rear_wheel_belt_y_range=(0.7, 1.2),
            ),
        )
        meshing_params = MeshingParams(
            defaults=MeshingDefaults(
                surface_max_aspect_ratio=10,
                curvature_resolution_angle=15 * u.deg,
                geometry_accuracy=1e-2,
                boundary_layer_first_layer_thickness=1e-4,
                boundary_layer_growth_rate=1.2,
                planar_face_tolerance=1e-3,
            ),
            volume_zones=[wind_tunnel],
        )
        param = SimulationParams(meshing=meshing_params)

    translated = get_volume_meshing_json(param, u.m)
    ref_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ref",
        "volume_meshing",
        "ref_param_to_json_wind_tunnel.json",
    )
    with open(ref_path, "r") as fh:
        ref_dict = json.load(fh)
    assert compare_values(translated, ref_dict)


def test_sliding_interface_tolerance_meshing_params(get_surface_mesh):
    """Test that sliding_interface_tolerance is translated correctly in MeshingParams."""
    with SI_unit_system:
        param = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=1e-3,
                    boundary_layer_growth_rate=1.2,
                    sliding_interface_tolerance=5e-3,
                ),
                volume_zones=[AutomatedFarfield()],
            ),
            private_attribute_asset_cache=AssetCache(use_inhouse_mesher=True),
        )
    translated = get_volume_meshing_json(param, get_surface_mesh.mesh_unit)
    assert "volume" in translated
    assert "slidingInterfaceTolerance" in translated["volume"]
    assert translated["volume"]["slidingInterfaceTolerance"] == 5e-3


def test_sliding_interface_tolerance_modular_workflow(get_surface_mesh):
    """Test that sliding_interface_tolerance is translated correctly in ModularMeshingWorkflow."""
    with SI_unit_system:
        param = SimulationParams(
            meshing=ModularMeshingWorkflow(
                volume_meshing=VolumeMeshingParams(
                    defaults=VolumeMeshingDefaults(
                        boundary_layer_first_layer_thickness=1e-3,
                        boundary_layer_growth_rate=1.2,
                    ),
                    sliding_interface_tolerance=2e-3,
                ),
                zones=[AutomatedFarfield()],
            ),
            private_attribute_asset_cache=AssetCache(use_inhouse_mesher=True),
        )
    translated = get_volume_meshing_json(param, get_surface_mesh.mesh_unit)
    assert "volume" in translated
    assert "slidingInterfaceTolerance" in translated["volume"]
    assert translated["volume"]["slidingInterfaceTolerance"] == 2e-3


def test_sliding_interface_tolerance_default_value(get_surface_mesh):
    """Test that default sliding_interface_tolerance value is used when not specified."""
    with SI_unit_system:
        param = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=1e-3,
                    boundary_layer_growth_rate=1.2,
                ),
                volume_zones=[AutomatedFarfield()],
            ),
            private_attribute_asset_cache=AssetCache(use_inhouse_mesher=True),
        )
    translated = get_volume_meshing_json(param, get_surface_mesh.mesh_unit)
    assert "volume" in translated
    assert "slidingInterfaceTolerance" in translated["volume"]
    # Default value is 1e-2 from DEFAULT_SLIDING_INTERFACE_TOLERANCE
    assert translated["volume"]["slidingInterfaceTolerance"] == 1e-2


def test_uniform_refinement_box_cylinder_axisymm_body(get_surface_mesh):
    """Test that Box, Cylinder, and AxisymmetricBody are correctly translated in UniformRefinement."""
    with SI_unit_system:
        cylinder = Cylinder(
            name="test_cylinder",
            outer_radius=1.0,
            height=2.0 * u.m,
            axis=(0, 0, 1),
            center=(0, 0, 0),
        )
        box = Box.from_principal_axes(
            name="test_box",
            center=(1, 2, 3),
            size=(2, 3, 4),
            axes=((1, 0, 0), (0, 1, 0)),
        )
        axisymmetric_body = AxisymmetricBody(
            name="test_cone",
            axis=(0, 1, 0),
            center=(5, 6, 7),
            profile_curve=[(0, 0), (0, 0.5), (1, 1), (1, 0)],
        )
        param = SimulationParams(
            meshing=MeshingParams(
                refinement_factor=1.0,
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=1e-5 * u.m,
                    boundary_layer_growth_rate=1.2,
                ),
                volume_zones=[AutomatedFarfield()],
                refinements=[
                    UniformRefinement(
                        entities=[cylinder, box, axisymmetric_body],
                        spacing=0.1 * u.m,
                    ),
                ],
            ),
            private_attribute_asset_cache=AssetCache(use_inhouse_mesher=True),
        )

    translated = get_volume_meshing_json(param, get_surface_mesh.mesh_unit)
    assert "refinement" in translated
    assert len(translated["refinement"]) == 3

    cylinder_ref = translated["refinement"][0]
    assert cylinder_ref["type"] == "cylinder"
    assert cylinder_ref["radius"] == 1.0
    assert cylinder_ref["length"] == 2.0
    assert cylinder_ref["axis"] == [0.0, 0.0, 1.0]
    assert cylinder_ref["center"] == [0.0, 0.0, 0.0]
    assert cylinder_ref["spacing"] == 0.1

    box_ref = translated["refinement"][1]
    assert box_ref["type"] == "box"
    assert box_ref["size"] == [2.0, 3.0, 4.0]
    assert box_ref["center"] == [1.0, 2.0, 3.0]
    assert box_ref["spacing"] == 0.1

    axisymm_ref = translated["refinement"][2]
    assert axisymm_ref["type"] == "Axisymmetric"
    assert axisymm_ref["axisOfRotation"] == [0.0, 1.0, 0.0]
    assert axisymm_ref["center"] == [5.0, 6.0, 7.0]
    assert axisymm_ref["profileCurve"] == [[0.0, 0.0], [0.0, 0.5], [1.0, 1.0], [1.0, 0.0]]
    assert axisymm_ref["spacing"] == 0.1


def test_windtunnel_ghost_surface_supported_in_volume_face_refinements(get_surface_mesh):
    with SI_unit_system:
        wind_tunnel = WindTunnelFarfield()
        param = SimulationParams(
            meshing=MeshingParams(
                refinement_factor=1.1,
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=1e-3,
                    boundary_layer_growth_rate=1.2,
                ),
                volume_zones=[wind_tunnel],
                refinements=[
                    BoundaryLayer(
                        entities=[wind_tunnel.floor],
                        first_layer_thickness=1e-3 * u.m,
                        growth_rate=1.2,
                    ),
                    PassiveSpacing(entities=[wind_tunnel.inlet], type="projected"),
                ],
            ),
            private_attribute_asset_cache=AssetCache(use_inhouse_mesher=True),
        )

    translated = get_volume_meshing_json(param, get_surface_mesh.mesh_unit)
    assert "faces" in translated
    assert translated["faces"]["windTunnelFloor"]["type"] == "aniso"
    assert translated["faces"]["windTunnelInlet"]["type"] == "projectAnisoSpacing"


def test_custom_volume_with_ghost_surface_farfield(get_surface_mesh):
    """GhostSurface(name='farfield') should be skipped from patches and force zone_name='farfield'."""
    auto_farfield = AutomatedFarfield()
    left1 = Surface(name="left1")
    right1 = Surface(name="right1")
    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=1e-4,
                    boundary_layer_growth_rate=1.2,
                ),
                volume_zones=[
                    CustomZones(
                        name="interior_zone",
                        entities=[
                            CustomVolume(
                                name="inner",
                                boundaries=[left1, right1],
                            ),
                        ],
                    ),
                    CustomZones(
                        name="exterior_zone",
                        entities=[
                            CustomVolume(
                                name="outer",
                                boundaries=[
                                    left1,
                                    right1,
                                    auto_farfield.farfield,
                                ],
                            ),
                        ],
                    ),
                    auto_farfield,
                ],
            ),
            private_attribute_asset_cache=AssetCache(use_inhouse_mesher=True),
        )

    translated = get_volume_meshing_json(params, get_surface_mesh.mesh_unit)
    assert "zones" in translated
    zones_by_name = {z["name"]: z for z in translated["zones"]}
    # Exterior CustomVolume should be renamed to "farfield" for the mesher
    assert "farfield" in zones_by_name
    # GhostSurface should not appear in patches
    assert "farfield" not in zones_by_name["farfield"]["patches"]
    assert sorted(zones_by_name["farfield"]["patches"]) == ["left1", "right1"]
    # Inner zone keeps its original name
    assert "inner" in zones_by_name
    assert sorted(zones_by_name["inner"]["patches"]) == ["left1", "right1"]
