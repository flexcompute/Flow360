import json
import os
from typing import Type, Union

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.meshing_param.edge_params import (
    AngleBasedRefinement,
    AspectRatioBasedRefinement,
    HeightBasedRefinement,
)
from flow360.component.simulation.meshing_param.face_params import (
    BoundaryLayer,
    SurfaceRefinement,
)
from flow360.component.simulation.meshing_param.params import (
    MeshingDefaults,
    MeshingParams,
    SurfaceEdgeRefinement,
)
from flow360.component.simulation.meshing_param.volume_params import (
    AutomatedFarfield,
    RotationVolume,
    UniformRefinement,
)
from flow360.component.simulation.models.material import SolidMaterial
from flow360.component.simulation.models.solver_numerics import TransitionModelSolver
from flow360.component.simulation.models.surface_models import (
    Freestream,
    HeatFlux,
    Inflow,
    Mach,
    MassFlowRate,
    Outflow,
    Periodic,
    Pressure,
    Rotational,
    SlipWall,
    SymmetryPlane,
    TotalPressure,
    Translational,
    Wall,
)
from flow360.component.simulation.models.turbulence_quantities import (
    TurbulenceQuantities,
)
from flow360.component.simulation.models.volume_models import (
    ActuatorDisk,
    AngularVelocity,
    BETDisk,
    Fluid,
    HeatEquationInitialCondition,
    NavierStokesInitialCondition,
    PorousMedium,
    Rotation,
    Solid,
)
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
    GenericReferenceCondition,
    ThermalState,
)
from flow360.component.simulation.outputs.output_entities import Point
from flow360.component.simulation.outputs.outputs import (
    AeroAcousticOutput,
    Isosurface,
    IsosurfaceOutput,
    Observer,
    ProbeOutput,
    Slice,
    SliceOutput,
    SurfaceIntegralOutput,
    SurfaceOutput,
    TimeAverageSurfaceOutput,
    TimeAverageVolumeOutput,
    VolumeOutput,
)
from flow360.component.simulation.primitives import (
    Box,
    Cylinder,
    Edge,
    GenericVolume,
    ReferenceGeometry,
    Surface,
    SurfacePair,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import (
    RampCFL,
    Steady,
    Unsteady,
)
from flow360.component.simulation.unit_system import (
    SI_unit_system,
    imperial_unit_system,
    u,
)
from flow360.component.simulation.user_defined_dynamics.user_defined_dynamics import (
    UserDefinedDynamic,
)

here = os.path.dirname(os.path.abspath(__file__))
version_postfix = "workbench-24.9"


data_folder = "data_v2"


def merge_dicts_recursively(dict1, dict2):
    """
    Recursively merges dict2 into dict1 with overwriting existing keys.

    Parameters
    ----------
    dict1 : dict
        The dictionary to be updated.
    dict2 : dict
        The dictionary to merge into dict1.

    Returns
    -------
    dict
        The merged dictionary.
    """
    for key, value in dict2.items():
        if key in dict1:
            if isinstance(dict1[key], dict) and isinstance(value, dict):
                merge_dicts_recursively(dict1[key], value)
            else:
                dict1[key] = value
        else:
            dict1[key] = value
    return dict1


def write_to_file(name, content):
    with open(name, "w") as outfile:
        outfile.write(content)


def write_schemas(type_obj: Type[Flow360BaseModel], folder_name, file_suffix=""):
    data = type_obj.model_json_schema()
    schema = json.dumps(data, indent=2)
    name = type_obj.__name__
    if name.startswith("_"):
        name = name[1:]
    os.makedirs(os.path.join(here, data_folder, folder_name), exist_ok=True)
    file_suffix_part = f"-{file_suffix}" if file_suffix else ""
    write_to_file(
        os.path.join(
            here,
            data_folder,
            folder_name,
            f"json-schema-{version_postfix}{file_suffix_part}.json",
        ),
        schema,
    )


def write_example(
    obj: Union[Flow360BaseModel, dict],
    folder_name,
    example_name,
    exclude_defaults=False,
    additional_fields: dict = {},
    exclude=None,
):
    if isinstance(obj, dict):
        data = obj
    elif isinstance(obj, Flow360BaseModel):
        data = obj.model_dump(exclude_defaults=exclude_defaults, exclude=exclude, exclude_none=True)
    else:
        raise ValueError("obj should be either dict or Flow360BaseModel")

    data = merge_dicts_recursively(data, additional_fields)
    data_json = json.dumps(data, indent=2)
    os.makedirs(os.path.join(here, data_folder, folder_name), exist_ok=True)
    write_to_file(
        os.path.join(here, data_folder, folder_name, f"{example_name}-{version_postfix}.json"),
        data_json,
    )


my_wall_surface = Surface(name="my_wall")
my_symm_plane = Surface(name="my_symmetry_plane")
my_slip_wall_surface = Surface(name="my_slip_wall")
my_inflow1 = Surface(name="my_inflow1")
my_inflow2 = Surface(name="my_inflow2")
my_outflow = Surface(name="my_outflow")
my_fs = Surface(name="my_free_stream")
edge = Edge(name="edge1")

my_surface_pair = SurfacePair(pair=(my_wall_surface, my_slip_wall_surface))


with SI_unit_system:
    my_box = Box(
        name="my_box",
        center=(1.2, 2.3, 3.4) * u.m,
        size=(1.0, 2.0, 3.0) * u.m,
        axis_of_rotation=(0, 0, 1),
        angle_of_rotation=0 * u.degree,
    )
    my_cylinder_1 = Cylinder(
        name="my_cylinder-1",
        axis=(5, 0, 0),
        center=(1.2, 2.3, 3.4) * u.m,
        height=3.0 * u.m,
        inner_radius=3.0 * u.m,
        outer_radius=5.0 * u.m,
    )
    my_cylinder_2 = my_cylinder_1.copy(update={"name": "my_cylinder-2"})
    my_solid_zone = GenericVolume(
        name="my_cylinder-2",
    )
    meshing = MeshingParams(
        defaults=MeshingDefaults(
            boundary_layer_first_layer_thickness=0.001,
            boundary_layer_growth_rate=1.3,
            surface_max_edge_length=1,
            surface_edge_growth_rate=1.4,
            curvature_resolution_angle=16 * u.deg,
        ),
        refinement_factor=1.0,
        gap_treatment_strength=0.5,
        refinements=[
            UniformRefinement(entities=[my_box], spacing=0.1 * u.m),
            UniformRefinement(entities=[my_box, my_cylinder_2], spacing=0.1 * u.m),
            SurfaceEdgeRefinement(edges=[edge], method=AngleBasedRefinement(value=1 * u.deg)),
            SurfaceEdgeRefinement(edges=[edge], method=HeightBasedRefinement(value=1 * u.m)),
            SurfaceEdgeRefinement(edges=[edge], method=AspectRatioBasedRefinement(value=2)),
        ],
        volume_zones=[
            RotationVolume(
                entities=my_cylinder_1,
                spacing_axial=0.1 * u.m,
                spacing_radial=0.12 * u.m,
                spacing_circumferential=0.13 * u.m,
                enclosed_entities=[my_wall_surface],
            ),
            AutomatedFarfield(method="auto"),
        ],
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
            Rotation(volumes=[my_cylinder_1], spec=AngularVelocity(0.45 * u.rad / u.s)),
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
                initial_condition=HeatEquationInitialCondition(temperature="1"),
            ),
            Inflow(
                surfaces=[my_inflow1],
                total_temperature=300 * u.K,
                spec=TotalPressure(value=123 * u.Pa),
                turbulence_quantities=TurbulenceQuantities(
                    turbulent_kinetic_energy=123, specific_dissipation_rate=1e3
                ),
            ),
            Inflow(
                surfaces=[my_inflow2],
                total_temperature=300 * u.K,
                spec=MassFlowRate(value=123 * u.lb / u.s),
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
        outputs=[
            ProbeOutput(
                name="my_probe",
                entities=[Point(name="-asf124", location=[1, 2, 3])],
                output_fields=["Cp"],
            ),
            SliceOutput(
                slices=[Slice(name="my_slice", normal=(0, 0, 1), origin=(0, 0, 0))],
                output_fields=["Cp"],
            ),
        ],
    )

write_example(param, "simulation_params", "example-1")

with SI_unit_system:
    meshing = MeshingParams(
        refinements=[
            BoundaryLayer(faces=[Surface(name="wing")], first_layer_thickness=0.001),
            SurfaceRefinement(entities=[Surface(name="wing")], max_edge_length=15 * u.cm),
        ],
    )
    param = SimulationParams(
        meshing=meshing,
        reference_geometry=ReferenceGeometry(
            moment_center=(1, 2, 3), moment_length=1.0 * u.m, area=1.0 * u.cm**2
        ),
        operating_condition=AerospaceCondition(velocity_magnitude=100),
        models=[
            Fluid(),
            Wall(
                entities=[
                    Surface(name="fluid/rightWing"),
                    Surface(name="fluid/leftWing"),
                    Surface(name="fluid/fuselage"),
                ],
            ),
            Freestream(entities=[Surface(name="fluid/farfield")]),
        ],
        time_stepping=Steady(max_steps=700),
    )

write_example(param, "simulation_params", "example-2")


###################### reference_geometry ######################
write_schemas(ReferenceGeometry, "reference_geometry")
with SI_unit_system:
    g = ReferenceGeometry(moment_center=(1, 2, 3), moment_length=(4, 5, 6), area=10)
    write_example(g, "reference_geometry", "example-1")


###################### meshing ######################
write_schemas(MeshingParams, "meshing")
with SI_unit_system:
    write_example(meshing, "meshing", "example-1")


write_schemas(UniformRefinement, "meshing", "uniform_refinement")
with SI_unit_system:
    ur = UniformRefinement(entities=[my_box, my_cylinder_1], spacing=0.1 * u.m)
    write_example(ur, "meshing", "uniform_refinement")


###################### operating_condition ######################
write_schemas(AerospaceCondition, "operating_condition", "aerospace_condition")
write_schemas(GenericReferenceCondition, "operating_condition", "generic_reference_condition")

with SI_unit_system:
    ac = AerospaceCondition(
        velocity_magnitude=1 * u.m / u.s,
        alpha=1 * u.deg,
        thermal_state=ThermalState(temperature=100 * u.K, density=2),
    )
write_example(
    ac,
    "operating_condition",
    "example-1",
    exclude_defaults=True,
    additional_fields=dict(
        type_name="AerospaceCondition", thermal_state=dict(type_name="ThermalState")
    ),
)

with SI_unit_system:
    ac = AerospaceCondition(
        velocity_magnitude=1 * u.m / u.s,
        thermal_state=ThermalState.from_standard_atmosphere(altitude=1000, temperature_offset=-1),
    )
write_example(
    ac,
    "operating_condition",
    "example-2",
    exclude_defaults=True,
    additional_fields=dict(
        type_name="AerospaceCondition", thermal_state=dict(type_name="ThermalState")
    ),
)

with SI_unit_system:
    ac = AerospaceCondition.from_mach(
        mach=0.8,
        alpha=1 * u.deg,
        thermal_state=ThermalState(temperature=100 * u.K, density=2),
    )
write_example(
    ac,
    "operating_condition",
    "example-3",
    exclude_defaults=True,
    exclude=dict(velocity_magnitude=True, private_attribute_input_cache=dict(thermal_state=True)),
    additional_fields=dict(
        type_name="AerospaceCondition", thermal_state=dict(type_name="ThermalState")
    ),
)

with SI_unit_system:
    ac = AerospaceCondition.from_mach(
        mach=0.8,
        alpha=1 * u.deg,
        thermal_state=ThermalState.from_standard_atmosphere(altitude=1000, temperature_offset=-1),
    )
write_example(
    ac,
    "operating_condition",
    "example-4",
    exclude_defaults=True,
    exclude=dict(velocity_magnitude=True, private_attribute_input_cache=dict(thermal_state=True)),
    additional_fields=dict(
        type_name="AerospaceCondition", thermal_state=dict(type_name="ThermalState")
    ),
)

###################### models  ######################
write_schemas(Fluid, "models", "fluid")
with imperial_unit_system:
    fluid_model = Fluid(
        transition_model_solver=TransitionModelSolver(),
        initial_condition=NavierStokesInitialCondition(rho="1;", u="1;", v="1;", w="1;", p="1;"),
    )
write_example(fluid_model, "models", "fluid")


write_schemas(Solid, "models", "solid")
with imperial_unit_system:
    solid_model = Solid(
        volumes=[my_solid_zone],
        material=SolidMaterial(
            name="abc",
            thermal_conductivity=1.0 * u.W / u.m / u.K,
            specific_heat_capacity=1.0 * u.J / u.kg / u.K,
            density=1.0 * u.kg / u.m**3,
        ),
        volumetric_heat_source=123 * u.lb / u.s**3 / u.ft,
    )
write_example(solid_model, "models", "solid")

write_schemas(Rotation, "models", "rotation")
rotation_model = Rotation(
    volumes=[my_cylinder_1],
    spec=AngularVelocity(0.45 * u.deg / u.s),
    parent_volume=GenericVolume(name="outter_volume"),
)
write_example(rotation_model, "models", "rotation")


write_schemas(BETDisk, "models", "bet_disk")
write_schemas(ActuatorDisk, "models", "actuator_disk")


write_schemas(PorousMedium, "models", "porouse_medium")
porous_model = PorousMedium(
    volumes=[my_box],
    darcy_coefficient=(0.1, 2, 1.0) / u.cm / u.m,
    forchheimer_coefficient=(0.1, 2, 1.0) / u.ft,
    volumetric_heat_source=123 * u.lb / u.s**3 / u.ft,
)
write_example(porous_model, "models", "porouse_medium")


write_schemas(Wall, "models", "wall")
my_wall = Wall(
    entities=[my_wall_surface],
    use_wall_function=True,
    velocity=(1.0, 1.2, 2.4) * u.ft / u.s,
    heat_spec=HeatFlux(1.0 * u.W / u.m**2),
)
write_example(my_wall, "models", "wall")

write_schemas(SlipWall, "models", "slip_wall")
my_wall = SlipWall(entities=[my_slip_wall_surface])
write_example(my_wall, "models", "slip_wall")


write_schemas(Freestream, "models", "freestream")
my_fs_surface = Freestream(entities=[my_fs], velocity=("1", "2", "0"))
write_example(my_fs_surface, "models", "freestream")


write_schemas(Outflow, "models", "outflow")
with imperial_unit_system:
    my_outflow_obj = Outflow(entities=[my_outflow], spec=Pressure(1))
write_example(my_outflow_obj, "models", "outflow-Pressure")

with imperial_unit_system:
    my_outflow_obj = Outflow(entities=[my_outflow], spec=MassFlowRate(value=1))
write_example(my_outflow_obj, "models", "outflow-MassFlowRate")

my_outflow_obj = Outflow(entities=[my_outflow], spec=Mach(1))
write_example(my_outflow_obj, "models", "outflow-Mach")


write_schemas(Inflow, "models", "inflow")
with imperial_unit_system:
    my_inflow_surface_1 = Inflow(
        surfaces=[my_inflow1],
        total_temperature=300 * u.K,
        spec=TotalPressure(value=123 * u.Pa),
        turbulence_quantities=TurbulenceQuantities(
            turbulent_kinetic_energy=123, specific_dissipation_rate=1e3
        ),
    )
write_example(my_inflow_surface_1, "models", "inflow-TotalPressure")

with imperial_unit_system:
    my_inflow_surface_1 = Inflow(
        surfaces=[my_inflow1],
        total_temperature=300 * u.K,
        spec=MassFlowRate(value=123),
        turbulence_quantities=TurbulenceQuantities(
            turbulent_kinetic_energy=123, specific_dissipation_rate=1e3
        ),
    )
write_example(my_inflow_surface_1, "models", "inflow-MassFlowRate")


write_schemas(Periodic, "models", "periodic")
with imperial_unit_system:
    my_pbc = Periodic(entity_pairs=[my_surface_pair], spec=Translational())
write_example(my_pbc, "models", "periodic-Translational")

with imperial_unit_system:
    my_pbc = Periodic(entity_pairs=[my_surface_pair], spec=Rotational(axis_of_rotation=(0, 2, 0)))
write_example(my_pbc, "models", "periodic-Rotational")


write_schemas(SymmetryPlane, "models", "symmetry_plane")
with imperial_unit_system:
    my_symm = SymmetryPlane(entities=[my_symm_plane])
write_example(my_symm, "models", "symmetry_plane")


###################### time stepping  ######################
write_schemas(Unsteady, "time_stepping", file_suffix="unsteady")
with imperial_unit_system:
    unsteady_setting = Unsteady(
        max_pseudo_steps=123,
        steps=456,
        step_size=2 * 0.2,
    )
write_example(unsteady_setting, "time_stepping", "unsteady")

write_schemas(Steady, "time_stepping", file_suffix="steady")
with imperial_unit_system:
    steady_setting = Steady(max_steps=123, CFL=RampCFL(initial=1e-4, final=123, ramp_steps=21))
write_example(steady_setting, "time_stepping", "steady")

###################### outputs  ######################
write_schemas(SurfaceOutput, "outputs", file_suffix="SurfaceOutput")
with imperial_unit_system:
    setting = SurfaceOutput(
        entities=[my_wall_surface, my_inflow1],
        frequency=12,
        frequency_offset=1,
        output_format="paraview",
        output_fields=["nodeNormals", "yPlus"],
    )
write_example(setting, "outputs", "SurfaceOutput")

write_schemas(SurfaceOutput, "outputs", file_suffix="TimeAverageSurfaceOutput")
with imperial_unit_system:
    setting = TimeAverageSurfaceOutput(
        entities=[my_wall_surface, my_inflow1],
        frequency=12,
        frequency_offset=1,
        start_step=10,
        output_format="tecplot",
        write_single_file=True,
        output_fields=["nodeNormals", "yPlus"],
    )
write_example(setting, "outputs", "TimeAverageSurfaceOutput")

write_schemas(VolumeOutput, "outputs", file_suffix="VolumeOutput")
with imperial_unit_system:
    setting = VolumeOutput(
        frequency=12,
        frequency_offset=1,
        output_format="tecplot",
        output_fields=["Cp"],
    )
write_example(setting, "outputs", "VolumeOutput")

write_schemas(TimeAverageVolumeOutput, "outputs", file_suffix="TimeAverageVolumeOutput")
with imperial_unit_system:
    setting = TimeAverageVolumeOutput(
        frequency=12,
        frequency_offset=1,
        output_format="tecplot",
        output_fields=["Cp"],
        start_step=0,
    )
write_example(setting, "outputs", "TimeAverageVolumeOutput")

write_schemas(SliceOutput, "outputs", file_suffix="SliceOutput")
with imperial_unit_system:
    setting = SliceOutput(
        frequency=12,
        frequency_offset=1,
        output_format="tecplot",
        output_fields=["Cp"],
        slices=[
            Slice(name="my_first_slice", normal=(1, 0, 0), origin=(4, 4, 2)),
            Slice(name="my_second_slice", normal=(1, 0, 1), origin=(41, 14, 12)),
        ],
    )
write_example(setting, "outputs", "SliceOutput")

write_schemas(IsosurfaceOutput, "outputs", file_suffix="IsosurfaceOutput")
with imperial_unit_system:
    setting = IsosurfaceOutput(
        frequency=12,
        frequency_offset=1,
        output_format="tecplot",
        output_fields=["primitiveVars", "vorticity"],
        isosurfaces=[
            Isosurface(name="my_first_iso", field="qcriterion", iso_value=1234.0),
            Isosurface(name="my_second_iso", field="Cp", iso_value=12.0),
        ],
    )
write_example(setting, "outputs", "IsosurfaceOutput")

write_schemas(SurfaceIntegralOutput, "outputs", file_suffix="SurfaceIntegralOutput")
with imperial_unit_system:
    setting = SurfaceIntegralOutput(
        name="my_surface_integral",
        output_fields=["primitiveVars", "vorticity"],
        entities=[my_inflow1, my_inflow2],
    )
write_example(setting, "outputs", "SurfaceIntegralOutput")

write_schemas(ProbeOutput, "outputs", file_suffix="ProbeOutput")
with imperial_unit_system:
    setting = ProbeOutput(
        name="my_probe",
        output_fields=["primitiveVars", "vorticity"],
        entities=[
            Point(name="DoesNotMatter1", location=(1, 2, 3)),
            Point(name="DoesNotMatter2", location=(1, 2, 5)),
        ],
    )
write_example(setting, "outputs", "ProbeOutput")

write_schemas(AeroAcousticOutput, "outputs", file_suffix="AeroAcousticOutput")
with imperial_unit_system:
    setting = AeroAcousticOutput(
        write_per_surface_output=True,
        observers=[
            Observer(position=(1, 2, 3), group_name="1"),
            Observer(position=(2, 4, 6), group_name="1"),
        ],
    )
write_example(setting, "outputs", "AeroAcousticOutput")


###################### UDD  ######################
write_schemas(UserDefinedDynamic, "UDD", file_suffix="UserDefinedDynamic")
user_defined_dynamic = UserDefinedDynamic(
    name="fake",
    input_vars=["fake"],
    constants={"ff": 123},
    output_vars={"fake_out": "1+1"},
    state_vars_initial_value=["0+0"],
    update_law=["1-2"],
    input_boundary_patches=[my_wall_surface],
    output_target=my_cylinder_1,
)
write_example(user_defined_dynamic, "UDD", "UserDefinedDynamic")

###################### Multi Constructor example  ######################
# 1. Operating condition and thermal state both use from function:
example_dict = {
    "type_name": "AerospaceCondition",
    "private_attribute_constructor": "from_mach",
    "private_attribute_input_cache": {
        "alpha": {"value": 5.0, "units": "degree"},
        "beta": {"value": 0.0, "units": "degree"},
        "thermal_state": {
            "type_name": "ThermalState",
            "private_attribute_constructor": "from_standard_atmosphere",
            "private_attribute_input_cache": {
                "altitude": {"value": 1000.0, "units": "m"},
                "temperature_offset": {"value": 0.0, "units": "K"},
            },
        },
        "mach": 0.8,
    },
}
write_example(example_dict, "multi_constructor", "recursive_from_function.json")

# 2. Operating condition and thermal state both use default constructor:
example_dict = {
    "type_name": "AerospaceCondition",
    "private_attribute_constructor": "default",
    "private_attribute_input_cache": {},
    "alpha": {"value": 5.0, "units": "degree"},
    "beta": {"value": 0.0, "units": "degree"},
    "velocity_magnitude": {"value": 0.8, "units": "km/s"},
    "thermal_state": {
        "type_name": "ThermalState",
        "private_attribute_constructor": "default",
        "private_attribute_input_cache": {},
        "temperature": {"value": 288.15, "units": "K"},
        "density": {"value": 1.225, "units": "kg/m**3"},
        "material": {
            "type": "air",
            "name": "air",
            "dynamic_viscosity": {
                "reference_viscosity": {"value": 1.716e-05, "units": "Pa*s"},
                "reference_temperature": {"value": 273.15, "units": "K"},
                "effective_temperature": {"value": 110.4, "units": "K"},
            },
        },
    },
    "reference_velocity_magnitude": {"value": 100.0, "units": "m/s"},
}
write_example(example_dict, "multi_constructor", "default_constructor.json")

# 3. Operating condition use default constructor thermal state use from function
example_dict = {
    "type_name": "AerospaceCondition",
    "private_attribute_constructor": "default",
    "private_attribute_input_cache": {},
    "alpha": {"value": 5.0, "units": "degree"},
    "beta": {"value": 0.0, "units": "degree"},
    "velocity_magnitude": {"value": 0.8, "units": "km/s"},
    "thermal_state": {
        "type_name": "ThermalState",
        "private_attribute_constructor": "from_standard_atmosphere",
        "private_attribute_input_cache": {
            "altitude": {"value": 1000.0, "units": "m"},
            "temperature_offset": {"value": 0.0, "units": "K"},
        },
    },
    "reference_velocity_magnitude": {"value": 100.0, "units": "m/s"},
}
write_example(example_dict, "multi_constructor", "default_and_from_constructor.json")


# 4. entity list of boxes:
example_dict = {
    "entities": {
        "stored_entities": [
            {
                "private_attribute_entity_type_name": "Box",
                "name": "my_box_default",
                "private_attribute_zone_boundary_names": {"items": []},
                "type_name": "Box",
                "private_attribute_constructor": "default",
                "private_attribute_input_cache": {},
                "center": {"value": [1.0, 2.0, 3.0], "units": "m"},
                "size": {"value": [2.0, 2.0, 3.0], "units": "m"},
                "axis_of_rotation": [1.0, 0.0, 0.0],
                "angle_of_rotation": {"value": 20.0, "units": "degree"},
            },
            {
                "type_name": "Box",
                "private_attribute_constructor": "from_principal_axes",
                "private_attribute_input_cache": {
                    "axes": [[0.6, 0.8, 0.0], [0.8, -0.6, 0.0]],
                    "center": {"value": [7.0, 1.0, 2.0], "units": "m"},
                    "size": {"value": [2.0, 2.0, 3.0], "units": "m"},
                    "name": "my_box_from",
                },
            },
            {
                "private_attribute_entity_type_name": "Cylinder",
                "name": "my_cylinder_default",
                "private_attribute_zone_boundary_names": {"items": []},
                "axis": [0.0, 1.0, 0.0],
                "center": {"value": [1.0, 2.0, 3.0], "units": "m"},
                "height": {"value": 3.0, "units": "m"},
                "outer_radius": {"value": 2.0, "units": "m"},
            },
        ]
    }
}
write_example(example_dict, "multi_constructor", "box_mixed_with_cylinder.json")

#### Primitives ####
write_schemas(Box, "Primitives", file_suffix="Box")
write_schemas(GenericVolume, "Primitives", file_suffix="GenericVolume")
write_schemas(Surface, "Primitives", file_suffix="Surface")
write_schemas(Edge, "Primitives", file_suffix="Edge")
