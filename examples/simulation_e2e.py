import os

import flow360 as fl
from flow360.component.simulation.meshing_param.face_params import SurfaceRefinement, BoundaryLayer
from flow360.component.simulation.meshing_param.edge_params import (
    HeightBasedRefinement,
    ProjectAnisoSpacing,
    SurfaceEdgeRefinement,
)
from flow360.component.simulation.models.surface_models import Freestream, Wall, SlipWall
from flow360.component.simulation.models.volume_models import Fluid
from flow360.component.simulation.operating_condition import AerospaceCondition
from flow360.component.simulation.primitives import ReferenceGeometry, Surface
from flow360.component.simulation.services import (
    simulation_to_case_json,
    simulation_to_surface_meshing_json,
    simulation_to_volume_meshing_json,
)
from flow360.component.simulation.simulation_params import (
    MeshingParams,
    SimulationParams,
)
from flow360.component.simulation.time_stepping.time_stepping import Steady, RampCFL
from flow360.component.simulation.unit_system import SI_unit_system, u

from flow360.component.simulation.models.solver_numerics import (
    LinearSolver,
    NavierStokesSolver,
    SpalartAllmaras,
)
from flow360.component.simulation.outputs.outputs import (
    SliceOutput,
    SurfaceOutput,
    VolumeOutput,
    Slice,
)

# fl.UserConfig.set_profile("auto_test_1")
fl.Env.dev.active()

from flow360.component.geometry import Geometry
from flow360.examples import Airplane

from tests.simulation.conftest import AssetBase
from flow360.component.simulation.unit_system import LengthType, SI_unit_system
from flow360.component.simulation.primitives import Edge, Surface, Cylinder
from flow360.component.simulation.meshing_param.volume_params import UniformRefinement

SOLVER_VERSION = "workbench-24.6.0"


class TempGeometry(AssetBase):
    """Mimicing the final Geometry class that can retrieve the metadata of the geometry."""

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
            self.internal_registry.register(Edge(name=zone_name))
        for surface_name in self._get_meta_data()["surfaces"]:
            self.internal_registry.register(Surface(name=surface_name))

    def __init__(self, file_name: str):
        super().__init__()
        self.fname = file_name
        self._populate_registry()


# Ref: [Automated Meshing]
# (https://docs.flexcompute.com/projects/flow360/en/latest/
# quickStart/WebUI_AutomatedMeshing/WebUI_AutomatedMeshing.html)
my_geometry = TempGeometry("om6wing.csm")
with SI_unit_system:
    base_cylinder = Cylinder(
        name="cylinder_1",
        outer_radius=1.1,
        height=2 * u.m,
        axis=(0, 1, 0),
        center=(0.7, -1.0, 0),
    )

    my_wall = Surface(name="1")
    my_symmetry_plane = Surface(name="2")
    my_freestream = Surface(name="3")

    meshing_params = MeshingParams(
        surface_layer_growth_rate=1.07,
        # refinement_factor=1.45,
        refinement_factor=1.1,
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
            SurfaceEdgeRefinement(
                entities=[my_geometry["*AirfoilEdge"]],
                method=ProjectAnisoSpacing(),
            ),
            # === For volume meshing
            UniformRefinement(
                entities=[base_cylinder],
                spacing=7.5 * u.cm,
            ),
            UniformRefinement(
                entities=[
                    base_cylinder.copy({"name": "cylinder_2", "outer_radius": 2.2 * u.m}),
                ],
                # spacing=10 * u.cm,
                spacing=20 * u.cm,
            ),
            UniformRefinement(
                entities=[
                    base_cylinder.copy({"name": "cylinder_3", "outer_radius": 3.3 * u.m}),
                ],
                # spacing=0.175,
                spacing=0.35,
            ),
            UniformRefinement(
                entities=[
                    base_cylinder.copy({"name": "cylinder_4", "outer_radius": 4.5 * u.m}),
                ],
                # spacing=225 * u.mm,
                spacing=450 * u.mm,
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
                spacing=600 * u.mm,
            ),
            BoundaryLayer(type="aniso", first_layer_thickness=1.35e-05 * u.m, growth_rate=1 + 0.04),
        ],
    )
    param = SimulationParams(
        meshing=meshing_params,
        reference_geometry=ReferenceGeometry(
            area=0.748844455929999,
            moment_length=0.6460682372650963,
            moment_center=(0, 0, 0),
        ),
        operating_condition=AerospaceCondition.from_mach(
            mach=0.84,
            alpha=3.06 * u.degree,
        ),
        models=[
            Fluid(
                navier_stokes_solver=NavierStokesSolver(
                    absolute_tolerance=1e-10,
                    linear_solver=LinearSolver(max_iterations=25),
                    kappa_MUSCL=-1.0,
                ),
                turbulence_model_solver=SpalartAllmaras(
                    absolute_tolerance=1e-8,
                    linear_solver=LinearSolver(max_iterations=15),
                    modeling_constants=None,
                ),
            ),
            Wall(surfaces=[my_wall]),
            SlipWall(entities=[my_symmetry_plane]),
            Freestream(entities=[my_freestream]),
        ],
        time_stepping=Steady(CFL=RampCFL()),
        outputs=[
            VolumeOutput(
                output_format="paraview",
                output_fields=[
                    "primitiveVars",
                    "residualNavierStokes",
                    "residualTurbulence",
                    "Mach",
                ],
            ),
            SliceOutput(
                slices=[
                    Slice(
                        name="sliceName_1",
                        slice_normal=(0, 1, 0),
                        slice_origin=(0, 0.56413, 0) * u.m,
                    )
                ],
                output_format="tecplot",
                output_fields=[
                    "primitiveVars",
                    "vorticity",
                    "T",
                    "s",
                    "Cp",
                    "mut",
                    "mutRatio",
                    "Mach",
                ],
            ),
            SurfaceOutput(
                entities=[my_wall, my_symmetry_plane, my_freestream],
                output_format="paraview",
                output_fields=["nuHat"],
            ),
        ],
    )

params_as_dict = param.model_dump()


prefix = "testing-workbench-integration-om6wing-csm"

# geometry
geometry_draft = Geometry.from_file(
    "om6wing.csm", name=f"{prefix}-geometry", solver_version=SOLVER_VERSION
)
geometry = geometry_draft.submit()
# print(geometry)

# surface mesh
surface_json, hash = simulation_to_surface_meshing_json(
    params_as_dict, "SI", {"value": 100.0, "units": "cm"}
)
params = fl.SurfaceMeshingParams(**surface_json)

surface_mesh_draft = fl.SurfaceMesh.create(
    geometry_id=geometry.id,
    params=params,
    name=f"{prefix}-surface-mesh",
    solver_version=SOLVER_VERSION,
)
surface_mesh = surface_mesh_draft.submit()

# print(surface_mesh)

# volume mesh
volume_json, hash = simulation_to_volume_meshing_json(
    params_as_dict, "SI", {"value": 100.0, "units": "cm"}
)
params = fl.VolumeMeshingParams(**volume_json)

volume_mesh_draft = fl.VolumeMesh.create(
    surface_mesh_id=surface_mesh.id,
    name=f"{prefix}-volume-mesh",
    params=params,
    solver_version=SOLVER_VERSION,
)
volume_mesh = volume_mesh_draft.submit()
# print(volume_mesh)

# case
case_json, hash = simulation_to_case_json(params_as_dict, "SI", {"value": 100.0, "units": "cm"})
params = fl.Flow360Params(**case_json, legacy_fallback=True)
case_draft = volume_mesh.create_case(f"{prefix}-case", params, solver_version=SOLVER_VERSION)
case = case_draft.submit()
