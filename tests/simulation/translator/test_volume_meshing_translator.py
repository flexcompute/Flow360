import json
import os

import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.meshing_param.face_params import (
    BoundaryLayer,
    PassiveSpacing,
)
from flow360.component.simulation.meshing_param.params import (
    MeshingDefaults,
    MeshingParams,
)
from flow360.component.simulation.meshing_param.volume_params import (
    AutomatedFarfield,
    AxisymmetricRefinement,
    RotationCylinder,
    RotationVolume,
    StructuredBoxRefinement,
    UniformRefinement,
    UserDefinedFarfield,
)
from flow360.component.simulation.primitives import (
    AxisymmetricBody,
    Box,
    CustomVolume,
    Cylinder,
    Surface,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.translator.volume_meshing_translator import (
    get_volume_meshing_json,
)
from flow360.component.simulation.unit_system import LengthType, SI_unit_system
from flow360.component.simulation.utils import model_attribute_unlock
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

        porous_medium = Box.from_principal_axes(
            name="porousRegion",
            center=(0, 1, 1),
            size=(1, 2, 1),
            axes=((2, 2, 0), (-2, 2, 0)),
        )

        param = SimulationParams(
            meshing=MeshingParams(
                refinement_factor=1.45,
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=1.35e-06 * u.m,
                    boundary_layer_growth_rate=1 + 0.04,
                ),
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
                volume_zones=[
                    CustomVolume(
                        name="custom_volume-1",
                        enforceTetrahedralElements=True,
                        boundaries=[
                            Surface(name="interface1"),
                            Surface(name="interface2"),
                        ],
                    ),
                    UserDefinedFarfield(domain_type="half_body_negative_y"),
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
                ],
            ),
            private_attribute_asset_cache=AssetCache(use_inhouse_mesher=True),
        )
    return param


def test_param_to_json(get_test_param, get_surface_mesh):
    translated = get_volume_meshing_json(get_test_param, get_surface_mesh.mesh_unit)
    ref_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ref",
        "volume_meshing",
        "ref_param_to_json_inhouse.json",
    )
    with open(ref_path, "r") as fh:
        ref_dict = json.load(fh)
    assert sorted(translated.items()) == sorted(ref_dict.items())


def test_user_defined_farfield(get_test_param, get_surface_mesh):
    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(boundary_layer_first_layer_thickness=100),
                volume_zones=[UserDefinedFarfield()],
            )
        )
    translated = get_volume_meshing_json(params, get_surface_mesh.mesh_unit)
    reference = {
        "refinementFactor": 1.0,
        "farfield": {"type": "user-defined"},
        "volume": {
            "firstLayerThickness": 100.0,
            "growthRate": 1.2,
            "gapTreatmentStrength": 0.0,
        },
        "faces": {},
    }
    assert sorted(translated.items()) == sorted(reference.items())


def test_param_to_json_legacy_mesher(get_test_param, get_surface_mesh):
    # Flip to legacy mesher
    with model_attribute_unlock(get_test_param.private_attribute_asset_cache, "use_inhouse_mesher"):
        get_test_param.private_attribute_asset_cache.use_inhouse_mesher = False
    translated = get_volume_meshing_json(get_test_param, get_surface_mesh.mesh_unit)
    ref_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ref",
        "volume_meshing",
        "ref_param_to_json_legacy.json",
    )
    with open(ref_path, "r") as fh:
        ref_dict = json.load(fh)
    assert sorted(translated.items()) == sorted(ref_dict.items())
