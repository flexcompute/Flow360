import json
import os

import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.models.solver_numerics import (
    LinearSolver,
    NavierStokesSolver,
    SpalartAllmaras,
)
from flow360.component.simulation.models.surface_models import (
    Freestream,
    SlipWall,
    Wall,
)
from flow360.component.simulation.models.volume_models import Fluid
from flow360.component.simulation.operating_condition import AerospaceCondition
from flow360.component.simulation.outputs.output_entities import Slice
from flow360.component.simulation.outputs.outputs import (
    SliceOutput,
    SurfaceOutput,
    VolumeOutput,
)
from flow360.component.simulation.primitives import ReferenceGeometry, Surface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import RampCFL, Steady
from flow360.component.simulation.translator.solver_translator import get_solver_json
from flow360.component.simulation.unit_system import SI_unit_system
from tests.simulation.translator.utils.xv15BETDisk_param_generator import (
    create_steady_airplane_param,
    create_steady_hover_param,
    create_unsteady_hover_param,
    create_unsteady_hover_UDD_param,
)
from tests.simulation.translator.utils.xv15BETDiskNestedRotation_param_generator import (
    create_nested_rotation_param,
    cylinder_inner,
    cylinder_middle,
)
from tests.utils import compare_values


@pytest.fixture()
def get_om6Wing_tutorial_param():
    my_wall = Surface(name="1")
    my_symmetry_plane = Surface(name="2")
    my_freestream = Surface(name="3")
    with SI_unit_system:
        param = SimulationParams(
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
    return param


def translate_and_compare(param, mesh_unit, ref_json_file: str):
    translated = get_solver_json(param, mesh_unit=mesh_unit)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref", ref_json_file)) as fh:
        ref_dict = json.load(fh)

    assert compare_values(ref_dict, translated)


def test_om6wing_tutorial(get_om6Wing_tutorial_param):
    translate_and_compare(
        get_om6Wing_tutorial_param, mesh_unit=0.8059 * u.m, ref_json_file="Flow360_om6Wing.json"
    )


##::  Test with local test cases
def test_xv15_bet_disk(
    create_steady_hover_param,
    create_steady_airplane_param,
    create_unsteady_hover_param,
    create_unsteady_hover_UDD_param,
):
    param = create_steady_hover_param
    translate_and_compare(
        param, mesh_unit=1 * u.inch, ref_json_file="Flow360_xv15_bet_disk_steady_hover.json"
    )

    param = create_steady_airplane_param
    translate_and_compare(
        param, mesh_unit=1 * u.inch, ref_json_file="Flow360_xv15_bet_disk_steady_airplane.json"
    )

    param = create_unsteady_hover_param
    translate_and_compare(
        param, mesh_unit=1 * u.inch, ref_json_file="Flow360_xv15_bet_disk_unsteady_hover.json"
    )

    param = create_unsteady_hover_UDD_param
    translate_and_compare(
        param, mesh_unit=1 * u.inch, ref_json_file="Flow360_xv15_bet_disk_unsteady_hover_UDD.json"
    )


def test_xv15_bet_disk_nested_rotation(
    create_nested_rotation_param, cylinder_inner, cylinder_middle
):
    param = create_nested_rotation_param
    translate_and_compare(
        param, mesh_unit=1 * u.inch, ref_json_file="Flow360_xv15_bet_disk_nested_rotation.json"
    )
