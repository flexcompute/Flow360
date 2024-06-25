import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.meshing_param.face_params import BoundaryLayer
from flow360.component.simulation.meshing_param.params import MeshingParams
from flow360.component.simulation.meshing_param.volume_params import UniformRefinement
from flow360.component.simulation.primitives import Cylinder, Surface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.translator.volume_meshing_translator import (
    get_volume_meshing_json,
)
from flow360.component.simulation.unit_system import LengthType, SI_unit_system
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
        param = SimulationParams(
            meshing=MeshingParams(
                refinement_factor=1.45,
                refinements=[
                    UniformRefinement(
                        entities=[base_cylinder],
                        spacing=7.5 * u.cm,
                    ),
                    UniformRefinement(
                        entities=[
                            base_cylinder.copy({"name": "cylinder_2", "outer_radius": 2.2 * u.m}),
                        ],
                        spacing=10 * u.cm,
                    ),
                    UniformRefinement(
                        entities=[
                            base_cylinder.copy({"name": "cylinder_3", "outer_radius": 3.3 * u.m}),
                        ],
                        spacing=0.175,
                    ),
                    UniformRefinement(
                        entities=[
                            base_cylinder.copy({"name": "cylinder_4", "outer_radius": 4.5 * u.m}),
                        ],
                        spacing=225 * u.mm,
                    ),
                    UniformRefinement(
                        entities=[
                            Cylinder(
                                name="outter_cylinder",
                                outer_radius=6.5,
                                height=14.5 * u.m,
                                axis=(-1, 0, 0),
                                center=(2, -1.0, 0),
                            )
                        ],
                        spacing=300 * u.mm,
                    ),
                    BoundaryLayer(
                        type="aniso", first_layer_thickness=1.35e-06 * u.m, growth_rate=1 + 0.04
                    ),
                ],
            )
        )
    return param


def test_param_to_json(get_test_param, get_surface_mesh):
    translated = get_volume_meshing_json(get_test_param, get_surface_mesh.mesh_unit)
    import json

    print("====TRANSLATED====\n", json.dumps(translated, indent=4))
    ref_dict = {
        "refinementFactor": 1.45,
        "refinement": [
            {
                "type": "cylinder",
                "radius": 1.1,
                "length": 2.0,
                "spacing": 0.075,
                "axis": [0, 1, 0],
                "center": [0.7, -1.0, 0],
            },
            {
                "type": "cylinder",
                "radius": 2.2,
                "length": 2.0,
                "spacing": 0.1,
                "axis": [0, 1, 0],
                "center": [0.7, -1.0, 0],
            },
            {
                "type": "cylinder",
                "radius": 3.3,
                "length": 2.0,
                "spacing": 0.175,
                "axis": [0, 1, 0],
                "center": [0.7, -1.0, 0],
            },
            {
                "type": "cylinder",
                "radius": 4.5,
                "length": 2.0,
                "spacing": 0.225,
                "axis": [0, 1, 0],
                "center": [0.7, -1.0, 0],
            },
            {
                "type": "cylinder",
                "radius": 6.5,
                "length": 14.5,
                "spacing": 0.3,
                "axis": [-1, 0, 0],
                "center": [2, -1.0, 0],
            },
        ],
        "volume": {"firstLayerThickness": 1.35e-06, "growthRate": 1.04},
    }

    assert sorted(translated.items()) == sorted(ref_dict.items())
