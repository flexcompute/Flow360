import flow360 as fl
import flow360.component.simulation.units as u
from flow360.component.project import Project
from flow360.component.simulation.meshing_param.edge_params import (
    HeightBasedRefinement,
    ProjectAnisoSpacing,
    SurfaceEdgeRefinement,
)
from flow360.component.simulation.meshing_param.face_params import SurfaceRefinement
from flow360.component.simulation.meshing_param.params import (
    MeshingDefaults,
    MeshingParams,
)
from flow360.component.simulation.meshing_param.volume_params import (
    RotationCylinder,
    AxisymmetricRefinement,
    AutomatedFarfield,
    UniformRefinement,
)
from flow360.component.simulation.models.solver_numerics import (
    NavierStokesSolver,
    SpalartAllmaras,
)
from flow360.component.simulation.models.surface_models import (
    Freestream,
    SlipWall,
    Wall,
)
from flow360.component.simulation.models.material import (
    Air,
    Sutherland,
)
from flow360.component.simulation.models.volume_models import (
    Fluid,
    BETDisk,
    BETDiskTwist,
    BETDiskChord,
    BETDiskSectionalPolar
)
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
    ThermalState,
)
from flow360.component.simulation.outputs.outputs import SurfaceOutput, VolumeOutput
from flow360.component.simulation.primitives import Cylinder, ReferenceGeometry
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import RampCFL, Steady
from flow360.component.simulation.unit_system import SI_unit_system
from flow360.examples import Tutorial_BETDisk
from examples.migration_guide.bet_disk import bet_disk_convert

"""
In this tutorial case we are looking at a 3-element airfoil, which is a cross-section of the NASA CRM-HL configuration. Documentation for this tutorial is available in the link below.

https://docs.flexcompute.com/projects/flow360/en/latest/tutorials/Multielement_Configuration/Multielement_Configuration.html
"""


fl.Env.preprod.active()

project = Project.from_file(Tutorial_BETDisk.geometry, name="Tutorial BETDisk from Python")
#project = Project.from_cloud(project_id="prj-edc9c40f-1de9-4a7d-ac64-122a483609dc")
geometry = project.geometry

# show face and edge groupings
geometry.show_available_groupings(verbose_mode=True)
geometry.group_faces_by_tag("faceName")
geometry.group_edges_by_tag("edgeName")

#ask about whether to define inner_radius since by default it is assigned None and not 0

with SI_unit_system:
    cylinder1 = Cylinder(
        name="cylinder1", axis=[1, 0, 0], center=[-2.0, 5.0, 0], outer_radius=4.0, inner_radius=0, height=0.6
    )
    cylinder2 = Cylinder(
        name="cylinder2", axis=[1, 0, 0], center=[0, 5, 0], outer_radius=4.1, inner_radius=0, height=5
    )
    BETDisks, Cylinders = bet_disk_convert(file="BET_tutorial_Flow360.json", save=True, omega_unit=u.flow360_angular_velocity_unit)
    farfield = AutomatedFarfield(name="farfield", method="auto")
    params = SimulationParams(
        meshing=MeshingParams(
            defaults=MeshingDefaults(
                surface_edge_growth_rate=1.2,
                surface_max_edge_length=0.5,
                curvature_resolution_angle=30 * u.deg,
                boundary_layer_growth_rate=1.15,
                boundary_layer_first_layer_thickness=1e-06,
            ),
            volume_zones=[farfield],
            refinements=[
                AxisymmetricRefinement(
                    name="BET_Disk",
                    spacing_axial=0.02,
                    spacing_radial=0.03,
                    spacing_circumferential=0.06,
                    entities=cylinder1,
                ),
                UniformRefinement(name="cylinder_refinement", spacing=0.1, entities=[cylinder2]),
                SurfaceRefinement(
                    name="tip",
                    max_edge_length=0.01,
                    faces=[
                        geometry["tip"],
                    ],
                ),
                SurfaceEdgeRefinement(
                    name="aniso",
                    method=HeightBasedRefinement(value=0.0003),
                    edges=[
                        geometry["wingTrailingEdge"],
                        geometry["wingLeadingEdge"],
                    ],
                ),
                SurfaceEdgeRefinement(
                    name="projectAnisoSpacing", method=ProjectAnisoSpacing(), edges=[
                        geometry["rootAirfoilEdge"],
                        geometry["tipAirfoilEdge"],
                    ]
                ),
            ],
        ),
        reference_geometry=ReferenceGeometry(
            moment_center=[0.375, 0, 0], moment_length=[1.26666666, 1.26666666, 1.26666666], area=12.5
        ),
        operating_condition=AerospaceCondition.from_mach(
            mach=0.182,
            alpha=5 * u.deg,
            beta=0 * u.deg,
            thermal_state=ThermalState(
                temperature=288.15,
                material=Air(
                    dynamic_viscosity=Sutherland(
                        reference_temperature=288.15,
                        reference_viscosity=4.29166e-08 * u.flow360_viscosity_unit,
                        effective_temperature=110.4,
                    )
                )
            ),
            reference_mach=0.54
        ),
        time_stepping=Steady(max_steps=10000, CFL=RampCFL(initial=1 , final=100, ramp_steps=2000)),
        outputs=[
            VolumeOutput(
                name="VolumeOutput",
                output_fields=[
                    "primitiveVars",
                    "betMetrics",
                    "qcriterion",
                ],
            ),
            SurfaceOutput(
                name="SurfaceOutput",
                surfaces=geometry["*"],
                output_fields=[
                    "primitiveVars",
                    "Cp",
                    "Cf",
                    "CfVec",
                ],
            ),
        ],
        models=[
            Wall(
                surfaces=[
                    geometry["wing"],
                    geometry["tip"],
                ],
                name="wall",
            ),
            Freestream(surfaces=farfield.farfield, name="Freestream"),
            SlipWall(surfaces=farfield.symmetry_planes, name="slipwall"),
            Fluid(
                navier_stokes_solver=NavierStokesSolver(
                    absolute_tolerance=1e-12,
                ),
                turbulence_model_solver=SpalartAllmaras(
                    absolute_tolerance=1e-10,
                    update_jacobian_frequency=1,
                    equation_evaluation_frequency=1,
                ),
            ),
            BETDisks[0],
        ],
    )


project.run_case(params=params, name="Case of tutorial BETDisk from Python")
