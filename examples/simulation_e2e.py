import os

import flow360 as fl
from flow360.component.simulation.services import (
    simulation_to_case_json,
    simulation_to_surface_meshing_json,
    simulation_to_volume_meshing_json,
)

from flow360.component.simulation.unit_system import SI_unit_system, u
from flow360.component.simulation.meshing_param.face_params import BoundaryLayer, SurfaceRefinement
from flow360.component.simulation.primitives import Surface, ReferenceGeometry
from flow360.component.simulation.simulation_params import SimulationParams, MeshingParams
from flow360.component.simulation.operating_condition import AerospaceCondition
from flow360.component.simulation.models.volume_models import Fluid
from flow360.component.simulation.models.surface_models import Wall, Freestream
from flow360.component.simulation.time_stepping.time_stepping import Steady

fl.UserConfig.set_profile('auto_test_1')
fl.Env.dev.active()

from flow360.component.geometry import Geometry
from flow360.examples import Airplane
SOLVER_VERSION = "workbench-24.6.0"


with SI_unit_system:
    meshing = MeshingParams(
        surface_layer_growth_rate=1.5,
        refinements=[
            BoundaryLayer(first_layer_thickness=0.001),
            SurfaceRefinement(
                        entities=[Surface(name="wing")],
                        max_edge_length=15 * u.cm,
                        curvature_resolution_angle=10 * u.deg,
                    )
        ]
    )
    param = SimulationParams(
        meshing=meshing,
        reference_geometry=ReferenceGeometry(
            moment_center=(1, 2, 3), moment_length=1.0 * u.m, area=1.0 * u.cm**2
        ),
        operating_condition=AerospaceCondition(
            velocity_magnitude=100
        ),
        models=[
            Fluid(),
            Wall(
                entities=[Surface(name="fluid/rightWing"), Surface(name="fluid/leftWing"), Surface(name="fluid/fuselage")],
            ),
            Freestream(
                entities=[Surface(name="fluid/farfield")]
            )
        ],
        time_stepping=Steady(max_steps=700)
    )

params_as_dict = param.model_dump()
surface_json, hash = simulation_to_surface_meshing_json(params_as_dict, "SI", {"value": 100.0, "units": "cm"})
print(surface_json)
volume_json, hash = simulation_to_volume_meshing_json(params_as_dict, "SI", {"value": 100.0, "units": "cm"})
print(volume_json)
case_json, hash = simulation_to_case_json(params_as_dict, "SI", {"value": 100.0, "units": "cm"})
print(case_json)


prefix = "testing-workbench-integration-airplane-csm"

# geometry
geometry_draft = Geometry.from_file(Airplane.geometry, name=f"{prefix}-geometry", solver_version=SOLVER_VERSION)
geometry = geometry_draft.submit()
print(geometry)

# surface mesh
params = fl.SurfaceMeshingParams(**surface_json)

surface_mesh_draft = fl.SurfaceMesh.create(
    geometry_id=geometry.id,
    params=params,
    name=f"{prefix}-surface-mesh",
    solver_version=SOLVER_VERSION,
)
surface_mesh = surface_mesh_draft.submit()

print(surface_mesh)

# volume mesh
params = fl.VolumeMeshingParams(**volume_json)

volume_mesh_draft = fl.VolumeMesh.create(
    surface_mesh_id=surface_mesh.id,
    name=f"{prefix}-volume-mesh",
    params=params,
    solver_version=SOLVER_VERSION,
)
volume_mesh = volume_mesh_draft.submit()
print(volume_mesh)

# case
case_json['turbulenceModelSolver'].pop('modelingConstants')
params = fl.Flow360Params(**case_json, legacy_fallback=True)
case_draft = volume_mesh.create_case(f"{prefix}-case", params, solver_version=SOLVER_VERSION)
case = case_draft.submit()