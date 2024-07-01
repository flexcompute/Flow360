import os

import flow360 as fl
from flow360.component.simulation.meshing_param.face_params import (
    BoundaryLayer,
    SurfaceRefinement,
)
from flow360.component.simulation.models.surface_models import Freestream, Wall
from flow360.component.simulation.models.volume_models import Fluid
from flow360.component.simulation.operating_condition import AerospaceCondition
from flow360.component.simulation.primitives import Edge, ReferenceGeometry, Surface
from flow360.component.simulation.services import (
    simulation_to_case_json,
    simulation_to_surface_meshing_json,
    simulation_to_volume_meshing_json,
)
from flow360.component.simulation.simulation_params import (
    MeshingParams,
    SimulationParams,
)
from flow360.component.simulation.time_stepping.time_stepping import Steady
from flow360.component.simulation.unit_system import LengthType, SI_unit_system, u

# fl.UserConfig.set_profile("auto_test_1")
fl.Env.dev.active()

from flow360.component.geometry import Geometry
from flow360.examples import Airplane

SOLVER_VERSION = "workbench-24.6.2"


##:: Hardcoded Geometry metadata ::##
from tests.simulation.conftest import AssetBase


class TempGeometry(AssetBase):
    """Mimicing the final Geometry class"""

    fname: str
    mesh_unit: LengthType.Positive

    def _get_meta_data(self):
        return {
            "edges": {
                "leadingEdge": {},
                "trailingEdge": {},
            },
            "surfaces": {"fuselage": {}, "rightWing": {}, "leftWing": {}},
            "mesh_unit": {"units": "cm", "value": 100.0},
        }

    def _populate_registry(self):
        self.mesh_unit = LengthType.validate(self._get_meta_data()["mesh_unit"])
        for zone_name in self._get_meta_data()["edges"]:
            self.internal_registry.register(Edge(name=zone_name))
        for surface_name in self._get_meta_data()["surfaces"]:
            self.internal_registry.register(Surface(name=surface_name))

    def __init__(self, file_name: str):
        super().__init__()
        self.fname = file_name
        self._populate_registry()


geometry_meta = TempGeometry("airplane.csm")

with SI_unit_system:
    meshing = MeshingParams(
        surface_layer_growth_rate=1.5,
        refinements=[
            BoundaryLayer(first_layer_thickness=0.002),
            SurfaceRefinement(
                entities=[geometry_meta["*Wing*"]],
                max_edge_length=20 * u.cm,
                curvature_resolution_angle=20 * u.deg,
            ),
        ],
    )
    my_wall_BC = Boundary(name="wings", entities=[geometry_meta["*Wing*"]])
    param = SimulationParams(
        meshing=meshing,
        reference_geometry=ReferenceGeometry(
            moment_center=(1, 2, 3), moment_length=1.0 * u.m, area=1.0 * u.cm**2
        ),
        operating_condition=AerospaceCondition(velocity_magnitude=100),
        models=[
            Fluid(),
            Wall(
                name="wall",
                entities=[
                    # Surface(name="rightWing"),
                    # Surface(name="leftWing"),
                    # Surface(name="fuselage"),
                    # geometry_meta["*Wing*"],
                    # geometry_meta["fuselage"],
                    my_wall_BC
                ],
            ),
            Freestream(entities=[Surface(name="farfield")]),  # To be replaced with farfield entity
        ],
        time_stepping=Steady(max_steps=10),
    )


prefix = "workbench-airplane"

# geometry
geometry_draft = Geometry.from_file(
    Airplane.geometry, name=f"{prefix}-geometry", solver_version=SOLVER_VERSION
)
geometry = geometry_draft.submit()
print(geometry)

# surface mesh
surface_json, hash = simulation_to_surface_meshing_json(
    param.model_dump(), "SI", geometry_meta.mesh_unit
)
surface_meshing_params = fl.SurfaceMeshingParams(**surface_json)

surface_mesh_draft = fl.SurfaceMesh.create(
    geometry_id=geometry.id,
    params=surface_meshing_params,
    name=f"{prefix}-surface-mesh",
    solver_version=SOLVER_VERSION,
)
surface_mesh = surface_mesh_draft.submit()

print(surface_mesh)

# volume mesh
volume_json, hash = simulation_to_volume_meshing_json(
    param.model_dump(), "SI", geometry_meta.mesh_unit
)
volume_meshing_params = fl.VolumeMeshingParams(**volume_json)

volume_mesh_draft = fl.VolumeMesh.create(
    surface_mesh_id=surface_mesh.id,
    name=f"{prefix}-volume-mesh",
    params=volume_meshing_params,
    solver_version=SOLVER_VERSION,
)
volume_mesh = volume_mesh_draft.submit()
print(volume_mesh)

# case
volume_mesh = fl.VolumeMesh.from_cloud("bbc518c2-c618-45de-9fe5-ac5fd570472d")
case_json, hash = simulation_to_case_json(param.model_dump(), "SI", geometry_meta.mesh_unit)
params = fl.Flow360Params(**case_json, legacy_fallback=True)
case_draft = volume_mesh.create_case(f"{prefix}-case", params, solver_version=SOLVER_VERSION)
case = case_draft.submit()
