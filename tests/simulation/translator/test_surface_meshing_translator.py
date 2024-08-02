import json
import os

import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.meshing_param.edge_params import (
    AngleBasedRefinement,
    HeightBasedRefinement,
    ProjectAnisoSpacing,
    SurfaceEdgeRefinement,
)
from flow360.component.simulation.meshing_param.face_params import SurfaceRefinement
from flow360.component.simulation.meshing_param.params import MeshingParams
from flow360.component.simulation.primitives import Edge, Surface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.translator.surface_meshing_translator import (
    get_surface_meshing_json,
)
from flow360.component.simulation.unit_system import (
    LengthType,
    SI_unit_system,
    imperial_unit_system,
)
from tests.simulation.conftest import AssetBase
from tests.utils import compare_values


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
        elif self.fname == "geometry.egads":
            return {
                "surfaces": {
                    "Outer_Wing_mirrored": {},
                    "Stab_mirrored": {},
                    "Outer_Wing": {},
                    "Fuselage_H": {},
                    "Inner_Wing": {},
                    "Fin": {},
                    "Inner_Wing_mirrored": {},
                    "Stab": {},
                    "Fuselage_H_mirrored": {},
                    "Fuselage_V": {},
                },
                "mesh_unit": {"units": "m", "value": 1.0},
            }
        elif self.fname == "rotor.csm":
            return {
                "surfaces": {
                    "hub": {},
                    "blade": {},
                    "tip": {},
                },
                "edges": {
                    "leadingEdge": {},
                    "trailingEdge": {},
                    "tipEdge": {},
                    "bladeSplitEdge": {},
                    "hubCircle": {},
                    "hubSplitEdge": {},
                    "junctionEdge": {},
                },
                "mesh_unit": {"units": "inch", "value": 1.0},
            }
        else:
            raise ValueError("Invalid file name")

    def _populate_registry(self):
        self.mesh_unit = LengthType.validate(self._get_meta_data()["mesh_unit"])
        for zone_name in self._get_meta_data()["edges"] if "edges" in self._get_meta_data() else []:
            self.internal_registry.register(Edge(name=zone_name))
        for surface_name in (
            self._get_meta_data()["surfaces"] if "surfaces" in self._get_meta_data() else []
        ):
            self.internal_registry.register(Surface(name=surface_name))

    def __init__(self, file_name: str):
        super().__init__()
        self.fname = file_name
        self._populate_registry()


@pytest.fixture()
def om6wing_tutorial_global_plus_local_override():
    my_geometry = TempGeometry("om6wing.csm")
    with SI_unit_system:
        param = SimulationParams(
            meshing=MeshingParams(
                surface_layer_growth_rate=1.07,
                refinements=[
                    SurfaceRefinement(
                        max_edge_length=15 * u.cm,
                        curvature_resolution_angle=10 * u.deg,
                    ),
                    SurfaceRefinement(
                        entities=[my_geometry["wing"]],
                        max_edge_length=15 * u.cm,
                    ),
                    SurfaceEdgeRefinement(
                        entities=[my_geometry["wing*Edge"]],
                        method=HeightBasedRefinement(value=3e-2 * u.cm),
                    ),
                    SurfaceEdgeRefinement(
                        entities=[my_geometry["*AirfoilEdge"]],
                        method=ProjectAnisoSpacing(),
                    ),
                ],
            )
        )
    return param


@pytest.fixture()
def get_om6wing_geometry():
    return TempGeometry("om6wing.csm")


@pytest.fixture()
def om6wing_tutorial_global_only():
    my_geometry = TempGeometry("om6wing.csm")
    with SI_unit_system:
        param = SimulationParams(
            meshing=MeshingParams(
                surface_layer_growth_rate=1.07,
                refinements=[
                    SurfaceRefinement(
                        max_edge_length=15 * u.cm,
                        curvature_resolution_angle=10 * u.deg,
                    ),
                    SurfaceEdgeRefinement(
                        entities=[my_geometry["wing*Edge"]],
                        method=HeightBasedRefinement(value=3e-2 * u.cm),
                    ),
                    SurfaceEdgeRefinement(
                        entities=[my_geometry["*AirfoilEdge"]],
                        method=ProjectAnisoSpacing(),
                    ),
                ],
            )
        )
    return param


@pytest.fixture()
def get_airplane_geometry():
    return TempGeometry("geometry.egads")


@pytest.fixture()
def get_rotor_geometry():
    return TempGeometry("rotor.csm")


@pytest.fixture()
def airplane_surface_mesh():
    my_geometry = TempGeometry("geometry.egads")
    from numpy import pi

    with SI_unit_system:
        param = SimulationParams(
            meshing=MeshingParams(
                refinements=[
                    SurfaceRefinement(
                        max_edge_length=100 * u.cm,
                        curvature_resolution_angle=pi / 12 * u.rad,
                    ),
                    SurfaceRefinement(
                        entities=[my_geometry["Inner*"]],
                        max_edge_length=1.5 * u.m,
                    ),
                    SurfaceRefinement(
                        entities=[my_geometry["Outer*"]],
                        max_edge_length=700 * u.mm,
                    ),
                    SurfaceRefinement(
                        entities=[my_geometry["Stab*"]],
                        max_edge_length=0.5 * u.m,
                    ),
                    SurfaceRefinement(
                        entities=[my_geometry["Fin*"]],
                        max_edge_length=0.5 * u.m,
                    ),
                ],
            )
        )
    return param


@pytest.fixture()
def rotor_surface_mesh():
    rotor_geopmetry = TempGeometry("rotor.csm")
    with imperial_unit_system:
        param = SimulationParams(
            meshing=MeshingParams(
                surface_layer_growth_rate=1.2,
                refinements=[
                    SurfaceRefinement(
                        max_edge_length=10,
                        curvature_resolution_angle=15 * u.deg,
                    ),  # Global
                    SurfaceRefinement(
                        entities=[rotor_geopmetry["tip"]],
                        max_edge_length=0.1 * u.inch,
                    ),
                    SurfaceEdgeRefinement(
                        entities=[rotor_geopmetry["leadingEdge"]],
                        method=AngleBasedRefinement(value=1 * u.degree),
                    ),
                    SurfaceEdgeRefinement(
                        entities=[rotor_geopmetry["t*Edge"]],
                        method=HeightBasedRefinement(value=0.05 * u.inch),
                    ),
                    SurfaceEdgeRefinement(
                        entities=[rotor_geopmetry["bladeSplitEdge"]],
                        method=ProjectAnisoSpacing(),
                    ),
                    SurfaceEdgeRefinement(
                        entities=[rotor_geopmetry["hubCircle"]],
                        method=HeightBasedRefinement(value=0.01 * u.inch),
                    ),
                    SurfaceEdgeRefinement(
                        entities=[rotor_geopmetry["junctionEdge"]],
                        method=HeightBasedRefinement(value=0.01 * u.inch),
                    ),
                ],
            )
        )
    return param


def _translate_and_compare(param, mesh_unit, ref_json_file: str):
    translated = get_surface_meshing_json(param, mesh_unit=mesh_unit)
    with open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "ref/surface_meshing", ref_json_file
        )
    ) as fh:
        ref_dict = json.load(fh)

    compare_values(ref_dict, translated)


def test_om6wing_tutorial(
    get_om6wing_geometry, om6wing_tutorial_global_plus_local_override, om6wing_tutorial_global_only
):
    _translate_and_compare(
        om6wing_tutorial_global_plus_local_override,
        get_om6wing_geometry.mesh_unit,
        "om6wing_tutorial_ignore_global.json",
    )
    _translate_and_compare(
        om6wing_tutorial_global_only,
        get_om6wing_geometry.mesh_unit,
        "om6wing_tutorial_use_global.json",
    )


def test_airplane_surface_mesh(get_airplane_geometry, airplane_surface_mesh):
    _translate_and_compare(
        airplane_surface_mesh,
        get_airplane_geometry.mesh_unit,
        "airplane.json",
    )


def test_rotor_surface_mesh(get_rotor_geometry, rotor_surface_mesh):
    _translate_and_compare(
        rotor_surface_mesh,
        get_rotor_geometry.mesh_unit,
        "rotor.json",
    )
