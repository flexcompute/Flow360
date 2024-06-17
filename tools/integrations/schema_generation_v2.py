from typing import Type, Annotated, Union, List
import json
import os
import pydantic as pd

from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.primitives import ReferenceGeometry
from flow360.component.simulation.meshing_param.params import (
    MeshingParams, 
    SurfaceEdgeRefinement, 
)
from flow360.component.simulation.meshing_param.volume_params import (
    UniformRefinement,
)
from flow360.component.simulation.meshing_param.edge_params import AngleBasedRefinement
from flow360.component.simulation.primitives import Box, Cylinder, GenericVolume, Surface, Edge
from flow360.component.simulation.operating_condition import AerospaceCondition, ThermalState
from flow360.component.simulation.models.volume_models import (
    Fluid,
    RotatingReferenceFrame, 
    AngularVelocity,
    PorousMedium,
    Solid,
)
from flow360.component.simulation.models.material import SolidMaterial
from flow360.component.simulation.models.surface_models import (
    Wall, 
    HeatFlux, 
    SlipWall, 
    Inflow, 
    TotalPressure,
    MassFlowRate,
)
from flow360.component.simulation.models.turbulence_quantities import (
    TurbulenceQuantities,
)
from flow360.component.simulation.time_stepping.time_stepping import Unsteady
from flow360.component.simulation.user_defined_dynamics.user_defined_dynamics import UserDefinedDynamic
from flow360.component.simulation.unit_system import SI_unit_system, u


here = os.path.dirname(os.path.abspath(__file__))
version_postfix = "release-24.3"



data_folder = "data_v2"



def write_to_file(name, content):
    with open(name, "w") as outfile:
        outfile.write(content)



def write_schemas(type_obj: Type[Flow360BaseModel], folder_name):
    data = type_obj.model_json_schema()
    schema = json.dumps(data, indent=2)
    name = type_obj.__name__
    if name.startswith("_"):
        name = name[1:]
    os.makedirs(os.path.join(here, data_folder, folder_name), exist_ok=True)
    write_to_file(
        os.path.join(here, data_folder, folder_name, f"json-schema-{version_postfix}.json"), schema
    )


def write_example(obj: Flow360BaseModel, folder_name, example_name):
    data = obj.model_dump()
    data_json = json.dumps(data, indent=2)
    os.makedirs(os.path.join(here, data_folder, folder_name), exist_ok=True)
    write_to_file(
        os.path.join(here, data_folder, folder_name, f"{example_name}-{version_postfix}.json"), data_json
    )


my_wall_surface = Surface(name="my_wall")
my_slip_wall_surface = Surface(name="my_slip_wall")
my_inflow1 = Surface(name="my_inflow1")
my_inflow2 = Surface(name="my_inflow2")
edge = Edge(name="edge1")


with SI_unit_system:
    my_box = Box(
        name="my_box",
        center=(1.2, 2.3, 3.4) * u.m,
        size=(1.0, 2.0, 3.0) * u.m,
        axes=((3, 4, 0), (1, 0, 0)),
    )
    my_cylinder_1 = Cylinder(
        name="my_cylinder-1",
        axis=(5, 0, 0),
        center=(1.2, 2.3, 3.4) * u.m,
        height=3.0 * u.m,
        inner_radius=3.0 * u.m,
        outer_radius=5.0 * u.m,
    )
    my_solid_zone = GenericVolume(
        name="my_cylinder-2",
    )
    meshing=MeshingParams(
            farfield="auto",
            refinement_factor=1.0,
            gap_treatment_strength=0.5,
            surface_layer_growth_rate=1.5,
            refinements=[
                UniformRefinement(entities=[my_box], spacing=0.1 * u.m),
                SurfaceEdgeRefinement(edges=[edge], method=AngleBasedRefinement(value=1*u.deg))
            ]
        )
    param = SimulationParams(
        meshing=meshing,
        reference_geometry=ReferenceGeometry(
            moment_center=(1, 2, 3), moment_length=1.0 * u.m, area=1.0 * u.cm**2
        ),
        operating_condition=AerospaceCondition(
            velocity_magnitude=200,
            alpha=30 * u.deg,
            beta=20 * u.deg,
            thermal_state=ThermalState(temperature=300 * u.K, density=1 * u.g / u.cm**3),
        ),
        models=[
            Fluid(),
            Wall(
                entities=[my_wall_surface],
                use_wall_function=True,
                velocity=(1.0, 1.2, 2.4) * u.ft / u.s,
                heat_spec=HeatFlux(1.0 * u.W / u.m**2),
            ),
            SlipWall(entities=[my_slip_wall_surface]),
            RotatingReferenceFrame(
                volumes=[my_cylinder_1], rotation=AngularVelocity(0.45 * u.rad / u.s)
            ),
            PorousMedium(
                volumes=[my_box],
                darcy_coefficient=(0.1, 2, 1.0) / u.cm / u.m,
                forchheimer_coefficient=(0.1, 2, 1.0) / u.ft,
                volumetric_heat_source=123 * u.lb / u.s**3 / u.ft,
            ),
            Solid(
                volumes=[my_solid_zone],
                material=SolidMaterial(
                    name="abc",
                    thermal_conductivity=1.0 * u.W / u.m / u.K,
                    specific_heat_capacity=1.0 * u.J / u.kg / u.K,
                    density=1.0 * u.kg / u.m**3,
                ),
            ),
            Inflow(
                surfaces=[my_inflow1],
                total_temperature=300 * u.K,
                spec=TotalPressure(123 * u.Pa),
                turbulence_quantities=TurbulenceQuantities(
                    turbulent_kinetic_energy=123, specific_dissipation_rate=1e3
                ),
            ),
            Inflow(
                surfaces=[my_inflow2],
                total_temperature=300 * u.K,
                spec=MassFlowRate(123 * u.lb / u.s),
            ),
        ],
        time_stepping=Unsteady(step_size=2 * 0.2 * u.s, steps=123),
        user_defined_dynamics=[
            UserDefinedDynamic(
                name="fake",
                input_vars=["fake"],
                constants={"ff": 123},
                state_vars_initial_value=["fake"],
                update_law=["fake"],
            )
        ],
    )

write_example(param, 'simulation_params', 'example-1')

















write_schemas(ReferenceGeometry, "geometry")
with SI_unit_system:
    g = ReferenceGeometry(moment_center=(1,2,3), moment_length=(4,5,6), area=10)
    write_example(g, 'geometry', 'example-1')


write_schemas(MeshingParams, "meshing")
with SI_unit_system:
    write_example(meshing, 'meshing', 'example-1')


write_schemas(AerospaceCondition)