import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.meshing_param.edge_params import (
    HeightBasedRefinement,
    SurfaceEdgeRefinement,
)
from flow360.component.simulation.meshing_param.face_params import SurfaceRefinement
from flow360.component.simulation.meshing_param.params import MeshingParams
from flow360.component.simulation.primitives import Edge, Surface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.translator.surface_mesh_translator import (
    get_surface_mesh_json,
)
from flow360.component.simulation.unit_system import LengthType, SI_unit_system
from tests.simulation.conftest import AssetBase


class TempGeometry(AssetBase):
    """Mimicing the final VolumeMesh class"""

    fname: str
    mesh_unit: LengthType.Positive

    def _get_meta_data(self):
        if self.fname == "om6wing.csm":
            return {
                "edges": {
                    "wingLeadingEdge": {},
                    "wingTrailingEdge": {},
                    "rootAirfoilEdge": {},
                    "tipAirfoilEdge": {},
                },
                "surfaces": {"wing": {}},
                "mesh_unit": {"units": "m", "value": 1.0},
            }
        else:
            raise ValueError("Invalid file name")

    def _populate_registry(self):
        self.mesh_unit = LengthType.validate(self._get_meta_data()["mesh_unit"])
        for zone_name in self._get_meta_data()["edges"]:
            self._registry.register(Edge(name=zone_name))
        for surface_name in self._get_meta_data()["surfaces"]:
            self._registry.register(Surface(name=surface_name))

    def __init__(self, file_name: str):
        super().__init__()
        self.fname = file_name
        self._populate_registry()


@pytest.fixture()
def get_geometry():
    return TempGeometry("om6wing.csm")


@pytest.fixture()
def get_test_param():
    my_geometry = TempGeometry("om6wing.csm")
    with SI_unit_system:
        param = SimulationParams(
            meshing=MeshingParams(
                surface_layer_growth_rate=1.07,
                refinements=[
                    SurfaceRefinement(
                        entities=[my_geometry["wing"]],
                        max_edge_length=15 * u.cm,
                        curvature_resolution_angle=10 * u.deg,
                    ),
                    SurfaceEdgeRefinement(
                        entities=[my_geometry["wing*Edge"]],
                        method=HeightBasedRefinement(value=3e-2 * u.cm),
                    ),
                ],
            )
        )
    return param


def test_param_to_json(get_test_param, get_geometry):
    translated = get_surface_mesh_json(get_test_param, get_geometry.mesh_unit)
    import json

    print(json.dumps(translated, indent=4))
