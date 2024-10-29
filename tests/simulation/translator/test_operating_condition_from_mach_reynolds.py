import unittest

import json
import os

import pytest

import flow360.component.simulation.units as u

from flow360.component.simulation.meshing_param.params import (
    MeshingDefaults,
    MeshingParams,
)
from flow360.component.simulation.meshing_param.volume_params import (
    AutomatedFarfield,
)
from flow360.component.simulation.models.solver_numerics import (
    NavierStokesSolver,
    LinearSolver,
    SpalartAllmaras
)
from flow360.component.simulation.primitives import (
    ReferenceGeometry,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import (
    Steady,
    RampCFL
)
from flow360.component.simulation.models.surface_models import (
    Freestream,
    SlipWall,
    Wall,
)
from flow360.component.simulation.models.volume_models import Fluid
from flow360.component.simulation.operating_condition.operating_condition import (
    operating_condition_from_mach_reynolds
)
from flow360.component.simulation.outputs.outputs import (
    SurfaceOutput,
    VolumeOutput,
)
from flow360.component.simulation.primitives import ReferenceGeometry, Surface
from flow360.component.simulation.time_stepping.time_stepping import RampCFL, Steady
from flow360.component.simulation.translator.solver_translator import get_solver_json
from flow360.component.simulation.unit_system import SI_unit_system, CGS_unit_system, imperial_unit_system

from tests.utils import compare_values

assertions = unittest.TestCase("__init__")

@pytest.fixture()
def get_2dcrm_tutorial_param():
    with imperial_unit_system:
        my_wall = Surface(name="1")
        my_symmetry_plane = Surface(name="2")
        my_freestream = Surface(name="3")
        farfield = AutomatedFarfield(name='farfield', method='quasi-3d')
        param = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    surface_edge_growth_rate=1.17,
                    surface_max_edge_length=1.1,
                    curvature_resolution_angle=12*u.deg,
                    boundary_layer_growth_rate=1.17,
                    boundary_layer_first_layer_thickness=1.8487111e-06
                ),
                refinement_factor=1.35,
                gap_treatment_strength=0.5,
                volume_zones=[farfield],
            ),
            reference_geometry=ReferenceGeometry(
                moment_center=[0.25,0.005,0],
                moment_length=[1,1,1],
                area=0.01
            ),
            operating_condition=operating_condition_from_mach_reynolds(
                mach=0.2,
                reynolds=5e+6,
                temperature=272.1,
                alpha=16*u.deg,
                beta=0*u.deg
            ),
            time_stepping=Steady(
                max_steps=3000,
                CFL=RampCFL(
                    initial=20,
                    final=300,
                    ramp_steps=500
                )
            ),
            models=[
                Wall(surfaces=[my_wall]),
                SlipWall(entities=[my_symmetry_plane]),
                Freestream(entities=[my_freestream]),
                Fluid(
                    navier_stokes_solver=NavierStokesSolver(
                        absolute_tolerance=1e-11,
                        relative_tolerance=1e-2,
                        linear_solver=LinearSolver(max_iterations=35),
                        kappa_MUSCL=0.33,
                        order_of_accuracy=2,
                        update_jacobian_frequency=4,
                        equation_evaluation_frequency=1
                    ), 
                    turbulence_model_solver=SpalartAllmaras(
                        absolute_tolerance=1e-10,
                        linear_solver=LinearSolver(max_iterations=25),
                        equation_evaluation_frequency=1
                    )
                )
            ],
            outputs=[
                VolumeOutput(
                    name='VolumeOutput',
                    output_fields=[
                        'primitiveVars',
                        'vorticity',
                        'residualNavierStokes',
                        'residualTurbulence',
                        'Cp',
                        'Mach',
                        'qcriterion',
                        'mut'
                        ]
                ),
                SurfaceOutput(
                    name='SurfaceOutput',
                    surfaces=[my_wall, my_symmetry_plane, my_freestream],
                    output_fields=[
                        'primitiveVars',
                        'Cp',
                        'Cf',
                        'CfVec',
                        'yPlus'
                    ]
                )
            ]
        )

    return param


def translate_and_compare(param, mesh_unit, ref_json_file: str, atol=1e-15, rtol=1e-10):
    translated = get_solver_json(param, mesh_unit=mesh_unit)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref", ref_json_file)) as fh:
        ref_dict = json.load(fh)
    print(">>> translated = ", translated)
    print("=== translated ===\n", json.dumps(translated, indent=4, sort_keys=True))
    print("=== ref_dict ===\n", json.dumps(ref_dict, indent=4, sort_keys=True))
    assert compare_values(ref_dict, translated, atol=atol, rtol=rtol)


mesh_unit = 1*u.m

#comparing against config json file
def test_2dcrm_tutorial(get_2dcrm_tutorial_param):
    translate_and_compare(
        get_2dcrm_tutorial_param, mesh_unit=1*u.m, ref_json_file="Flow360_tutorial_2dcrm.json"
    )


def test_operating_condition(get_2dcrm_tutorial_param):
    converted = get_2dcrm_tutorial_param.preprocess(mesh_unit=mesh_unit)
    #operating conditions
    assertions.assertAlmostEqual(converted.operating_condition.velocity_magnitude.value, 0.2)
    assertions.assertEqual(converted.operating_condition.thermal_state._mu_ref(), 1.4260047486509403e-09)
    assertions.assertEqual(converted.operating_condition.thermal_state.temperature, 272.1)
    assertions.assertEqual(converted.operating_condition.thermal_state.material.dynamic_viscosity.reference_viscosity.value, 4e-8)
    assertions.assertEqual(converted.operating_condition.thermal_state.material.dynamic_viscosity.effective_temperature, 110.4 * u.K)
    assertions.assertEqual(converted.operating_condition.thermal_state.material.get_dynamic_viscosity(converted.operating_condition.thermal_state.temperature), 4e-8)