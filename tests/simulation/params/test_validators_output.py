import json
import os
import re

import pydantic
import pytest

import flow360 as fl
import flow360.component.simulation.units as u


def assert_validation_error_contains(
    error: pydantic.ValidationError,
    expected_loc: tuple,
    expected_msg_contains: str,
):
    """Helper function to assert validation error properties for moving_statistic tests"""
    errors = error.errors()
    # Find the error with matching location
    matching_errors = [e for e in errors if e["loc"] == expected_loc]
    assert (
        len(matching_errors) == 1
    ), f"Expected 1 error at {expected_loc}, found {len(matching_errors)}"
    assert expected_msg_contains in matching_errors[0]["msg"], (
        f"Expected '{expected_msg_contains}' in error message, "
        f"but got: '{matching_errors[0]['msg']}'"
    )
    assert matching_errors[0]["type"] == "value_error"


from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.models.solver_numerics import (
    KOmegaSST,
    NoneSolver,
    SpalartAllmaras,
)
from flow360.component.simulation.models.surface_models import Wall
from flow360.component.simulation.models.volume_models import Fluid, PorousMedium
from flow360.component.simulation.outputs.output_entities import Point
from flow360.component.simulation.outputs.outputs import (
    AeroAcousticOutput,
    ForceDistributionOutput,
    Isosurface,
    IsosurfaceOutput,
    MovingStatistic,
    ProbeOutput,
    SurfaceIntegralOutput,
    SurfaceOutput,
    SurfaceProbeOutput,
    TimeAverageForceDistributionOutput,
    TimeAverageSurfaceOutput,
    VolumeOutput,
)
from flow360.component.simulation.primitives import ImportedSurface, Surface
from flow360.component.simulation.services import (
    ValidationCalledBy,
    clear_context,
    validate_model,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import Steady, Unsteady
from flow360.component.simulation.unit_system import (
    SI_unit_system,
    imperial_unit_system,
)
from flow360.component.simulation.user_code.core.types import UserVariable
from flow360.component.simulation.user_code.functions import math
from flow360.component.simulation.user_code.variables import solution
from flow360.component.simulation.validation.validation_context import (
    CASE,
    ParamsValidationInfo,
    ValidationContext,
)
from flow360.component.volume_mesh import VolumeMeshV2


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


@pytest.fixture()
def reset_context():
    clear_context()


def test_aeroacoustic_observer_unit_validator():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "All observer locations should have the same unit. But now it has both `cm` and `mm`."
        ),
    ):
        AeroAcousticOutput(
            name="test",
            observers=[
                fl.Observer(position=[0.2, 0.02, 0.03] * u.cm, group_name="0"),
                fl.Observer(position=[0.0001, 0.02, 0.03] * u.mm, group_name="1"),
            ],
        )


def test_unsteadiness_to_use_aero_acoustics():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "In `outputs`[1] AeroAcousticOutput:`AeroAcousticOutput` can only be activated with `Unsteady` simulation."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                models=[Fluid(turbulence_model_solver=NoneSolver())],
                outputs=[
                    IsosurfaceOutput(
                        name="iso",
                        entities=[Isosurface(name="tmp", field="mut", iso_value=1)],
                        output_fields=["Cp"],
                    ),
                    AeroAcousticOutput(
                        name="test",
                        observers=[
                            fl.Observer(position=[0.2, 0.02, 0.03] * u.mm, group_name="0"),
                            fl.Observer(position=[0.0001, 0.02, 0.03] * u.mm, group_name="1"),
                        ],
                    ),
                ],
                time_stepping=fl.Steady(),
            )


def test_aero_acoustics_observer_time_step_size():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "In `outputs`[0] AeroAcousticOutput: "
            "`observer_time_size` (0.05 s) is smaller than the time step size of CFD (0.1 s)."
        ),
    ):
        with SI_unit_system:
            SimulationParams(
                outputs=[
                    AeroAcousticOutput(
                        name="test",
                        observers=[
                            fl.Observer(position=[0.2, 0.02, 0.03] * u.mm, group_name="0"),
                            fl.Observer(position=[0.0001, 0.02, 0.03] * u.mm, group_name="1"),
                        ],
                        observer_time_step_size=0.05,
                    ),
                ],
                time_stepping=Unsteady(steps=1, step_size=0.1),
            )


def test_turbulence_enabled_output_fields():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "In `outputs`[0] IsosurfaceOutput: kOmega is not a valid output field when using turbulence model: None."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                models=[Fluid(turbulence_model_solver=NoneSolver())],
                outputs=[
                    IsosurfaceOutput(
                        name="iso",
                        entities=[Isosurface(name="tmp", field="mut", iso_value=1)],
                        output_fields=["kOmega"],
                    )
                ],
            )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "In `outputs`[0] IsosurfaceOutput: nuHat is not a valid iso field when using turbulence model: kOmegaSST."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                models=[Fluid(turbulence_model_solver=KOmegaSST())],
                outputs=[
                    IsosurfaceOutput(
                        name="iso",
                        entities=[Isosurface(name="tmp", field="nuHat", iso_value=1)],
                        output_fields=["Cp"],
                    )
                ],
            )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "In `outputs`[0] VolumeOutput: kOmega is not a valid output field when using turbulence model: SpalartAllmaras."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                models=[Fluid(turbulence_model_solver=SpalartAllmaras())],
                outputs=[VolumeOutput(output_fields=["kOmega"])],
            )


def test_transition_model_enabled_output_fields():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "In `outputs`[0] IsosurfaceOutput: solutionTransition is not a valid output field when transition model is not used."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                models=[Fluid(transition_model_solver=NoneSolver())],
                outputs=[
                    IsosurfaceOutput(
                        name="iso",
                        entities=[Isosurface(name="tmp", field="mut", iso_value=1)],
                        output_fields=["solutionTransition"],
                    )
                ],
            )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "In `outputs`[0] SurfaceProbeOutput: residualTransition is not a valid output field when transition model is not used."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                models=[Fluid(transition_model_solver=NoneSolver())],
                outputs=[
                    SurfaceProbeOutput(
                        name="probe_output",
                        probe_points=[Point(name="point_1", location=[1, 2, 3] * u.m)],
                        output_fields=["residualTransition"],
                        target_surfaces=[Surface(name="fluid/body")],
                    )
                ],
            )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "In `outputs`[0] VolumeOutput: linearResidualTransition is not a valid output field when transition model is not used."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                models=[Fluid(transition_model_solver=NoneSolver())],
                outputs=[VolumeOutput(output_fields=["linearResidualTransition"])],
            )


def test_surface_user_variables_in_output_fields():
    uv_surface1 = UserVariable(
        name="uv_surface1", value=math.dot(solution.velocity, solution.CfVec)
    )
    uv_surface2 = UserVariable(
        name="uv_surface2", value=solution.node_forces_per_unit_area[0] * solution.Cp * solution.Cf
    )

    with imperial_unit_system:
        SimulationParams(
            outputs=[
                SurfaceOutput(
                    entities=Surface(name="fluid/body"), output_fields=[uv_surface1, uv_surface2]
                )
            ],
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Variable `uv_surface1` cannot be used in `VolumeOutput` "
            + "since it contains Surface solver variable(s): solution.CfVec."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                outputs=[VolumeOutput(output_fields=[uv_surface1])],
            )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Variable `uv_surface2` cannot be used in `ProbeOutput` "
            + "since it contains Surface solver variable(s): "
            + "solution.Cf, solution.node_forces_per_unit_area."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                outputs=[
                    ProbeOutput(
                        name="probe_output",
                        probe_points=[Point(name="point_1", location=[1, 2, 3] * u.m)],
                        output_fields=[uv_surface2],
                    )
                ],
            )


def test_duplicate_surface_usage(mock_validation_context):
    my_var = UserVariable(name="my_var", value=solution.node_forces_per_unit_area[1])
    with mock_validation_context, pytest.raises(
        ValueError,
        match=re.escape(
            "The same surface `fluid/body` is used in multiple `SurfaceOutput`s. "
            "Please specify all settings for the same surface in one output."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                outputs=[
                    SurfaceOutput(entities=Surface(name="fluid/body"), output_fields=[my_var]),
                    SurfaceOutput(
                        entities=Surface(name="fluid/body"), output_fields=[solution.CfVec]
                    ),
                ],
            )

    with mock_validation_context, pytest.raises(
        ValueError,
        match=re.escape(
            "The same surface `fluid/body` is used in multiple `TimeAverageSurfaceOutput`s. "
            "Please specify all settings for the same surface in one output."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                outputs=[
                    TimeAverageSurfaceOutput(
                        entities=Surface(name="fluid/body"), output_fields=[my_var]
                    ),
                    TimeAverageSurfaceOutput(
                        entities=Surface(name="fluid/body"), output_fields=[solution.CfVec]
                    ),
                ],
                time_stepping=Unsteady(steps=10, step_size=1e-3),
            )

    with imperial_unit_system:
        SimulationParams(
            outputs=[
                SurfaceOutput(entities=Surface(name="fluid/body"), output_fields=[solution.CfVec]),
                TimeAverageSurfaceOutput(
                    entities=Surface(name="fluid/body"), output_fields=[solution.CfVec]
                ),
            ],
            time_stepping=Unsteady(steps=10, step_size=1e-3),
        )


def test_check_moving_statistic_applicability_steady_valid():
    """Test moving_statistic with steady simulation - valid case."""
    wall_1 = Wall(entities=Surface(name="fluid/wing"))

    # Valid: window_size=10 (becomes 100 steps) + start_step=100 (becomes 100) = 200 <= 5000
    with imperial_unit_system:
        SimulationParams(
            models=[Fluid(), wall_1],
            time_stepping=Steady(max_steps=5000),
            outputs=[
                fl.ForceOutput(
                    name="force_output",
                    models=[wall_1],
                    output_fields=["CL", "CD"],
                    moving_statistic=MovingStatistic(
                        method="mean", moving_window_size=10, start_step=100
                    ),
                )
            ],
        )

    # Valid: window_size=5 (becomes 50 steps) + start_step=50 (becomes 50) = 100 <= 1000
    with imperial_unit_system:
        SimulationParams(
            models=[Fluid(), wall_1],
            time_stepping=Steady(max_steps=1000),
            outputs=[
                ProbeOutput(
                    name="probe_output",
                    probe_points=[Point(name="point_1", location=[1, 2, 3] * u.m)],
                    output_fields=["Cp"],
                    moving_statistic=MovingStatistic(
                        method="standard_deviation", moving_window_size=5, start_step=50
                    ),
                )
            ],
        )


def test_check_moving_statistic_applicability_steady_invalid():
    """Test moving_statistic with steady simulation - invalid case."""
    wall_1 = Wall(entities=Surface(name="fluid/wing"))

    # Invalid: window_size=50 (becomes 500 steps) + start_step=4600 (becomes 4600) = 5100 > 5000
    with pytest.raises(pydantic.ValidationError) as exc_info:
        with imperial_unit_system:
            SimulationParams(
                models=[Fluid(), wall_1],
                time_stepping=Steady(max_steps=5000),
                outputs=[
                    fl.ForceOutput(
                        name="force_output",
                        models=[wall_1],
                        output_fields=["CL", "CD"],
                        moving_statistic=MovingStatistic(
                            method="mean", moving_window_size=50, start_step=4600
                        ),
                    )
                ],
            )

    assert_validation_error_contains(
        exc_info.value,
        expected_loc=("outputs", 0, "moving_statistic"),
        expected_msg_contains="`moving_statistic`'s moving_window_size + start_step exceeds "
        "the total number of steps in the simulation.",
    )

    # Invalid: window_size=20 (becomes 200 steps) + start_step=850 (becomes 850) = 1060 > 1000
    with pytest.raises(pydantic.ValidationError) as exc_info:
        with imperial_unit_system:
            SimulationParams(
                models=[Fluid(), wall_1],
                time_stepping=Steady(max_steps=1000),
                outputs=[
                    ProbeOutput(
                        name="probe_output",
                        probe_points=[Point(name="point_1", location=[1, 2, 3] * u.m)],
                        output_fields=["Cp"],
                        moving_statistic=MovingStatistic(
                            method="standard_deviation", moving_window_size=20, start_step=850
                        ),
                    )
                ],
            )

    assert_validation_error_contains(
        exc_info.value,
        expected_loc=("outputs", 0, "moving_statistic"),
        expected_msg_contains="`moving_statistic`'s moving_window_size + start_step exceeds "
        "the total number of steps in the simulation.",
    )


def test_check_moving_statistic_applicability_unsteady_valid():
    """Test moving_statistic with unsteady simulation - valid case."""
    wall_1 = Wall(entities=Surface(name="fluid/wing"))

    # Valid: window_size=100 + start_step=200 = 300 <= 1000
    with imperial_unit_system:
        SimulationParams(
            models=[Fluid(), wall_1],
            time_stepping=Unsteady(steps=1000, step_size=1e-3),
            outputs=[
                fl.ForceOutput(
                    name="force_output",
                    models=[wall_1],
                    output_fields=["CL", "CD"],
                    moving_statistic=MovingStatistic(
                        method="mean", moving_window_size=100, start_step=200
                    ),
                )
            ],
        )

    # Valid: window_size=50 + start_step=50 = 100 <= 500
    with imperial_unit_system:
        SimulationParams(
            models=[Fluid(), wall_1],
            time_stepping=Unsteady(steps=500, step_size=1e-3),
            outputs=[
                ProbeOutput(
                    name="probe_output",
                    probe_points=[Point(name="point_1", location=[1, 2, 3] * u.m)],
                    output_fields=["Cp"],
                    moving_statistic=MovingStatistic(
                        method="standard_deviation", moving_window_size=50, start_step=50
                    ),
                )
            ],
        )


def test_check_moving_statistic_applicability_unsteady_invalid():
    """Test moving_statistic with unsteady simulation - invalid case."""
    wall_1 = Wall(entities=Surface(name="fluid/wing"))

    # Invalid: window_size=500 + start_step=600 = 1100 > 1000
    with pytest.raises(pydantic.ValidationError) as exc_info:
        with imperial_unit_system:
            SimulationParams(
                models=[Fluid(), wall_1],
                time_stepping=Unsteady(steps=1000, step_size=1e-3),
                outputs=[
                    fl.ForceOutput(
                        name="force_output",
                        models=[wall_1],
                        output_fields=["CL", "CD"],
                        moving_statistic=MovingStatistic(
                            method="mean", moving_window_size=500, start_step=600
                        ),
                    )
                ],
            )

    assert_validation_error_contains(
        exc_info.value,
        expected_loc=("outputs", 0, "moving_statistic"),
        expected_msg_contains="`moving_statistic`'s moving_window_size + start_step exceeds "
        "the total number of steps in the simulation.",
    )

    # Invalid: window_size=200 + start_step=350 = 550 > 500
    with pytest.raises(pydantic.ValidationError) as exc_info:
        with imperial_unit_system:
            SimulationParams(
                models=[Fluid(), wall_1],
                time_stepping=Unsteady(steps=500, step_size=1e-3),
                outputs=[
                    ProbeOutput(
                        name="probe_output",
                        probe_points=[Point(name="point_1", location=[1, 2, 3] * u.m)],
                        output_fields=["Cp"],
                        moving_statistic=MovingStatistic(
                            method="standard_deviation", moving_window_size=200, start_step=350
                        ),
                    )
                ],
            )

    assert_validation_error_contains(
        exc_info.value,
        expected_loc=("outputs", 0, "moving_statistic"),
        expected_msg_contains="`moving_statistic`'s moving_window_size + start_step exceeds "
        "the total number of steps in the simulation.",
    )


def test_check_moving_statistic_applicability_steady_edge_cases():
    """Test moving_statistic with steady simulation - edge cases for rounding."""
    wall_1 = Wall(entities=Surface(name="fluid/wing"))

    # Edge case: start_step=47 rounds up to 50, window_size=10 becomes 100, total=150 <= 200
    with imperial_unit_system:
        SimulationParams(
            models=[Fluid(), wall_1],
            time_stepping=Steady(max_steps=200),
            outputs=[
                fl.ForceOutput(
                    name="force_output",
                    models=[wall_1],
                    output_fields=["CL", "CD"],
                    moving_statistic=MovingStatistic(
                        method="mean", moving_window_size=10, start_step=47
                    ),
                )
            ],
        )

    # Edge case: start_step=99 rounds up to 100, window_size=5 becomes 50, total=150 <= 200
    with imperial_unit_system:
        SimulationParams(
            models=[Fluid(), wall_1],
            time_stepping=Steady(max_steps=200),
            outputs=[
                ProbeOutput(
                    name="probe_output",
                    probe_points=[Point(name="point_1", location=[1, 2, 3] * u.m)],
                    output_fields=["Cp"],
                    moving_statistic=MovingStatistic(
                        method="standard_deviation", moving_window_size=5, start_step=99
                    ),
                )
            ],
        )

    # Edge case: start_step=100 (already multiple of 10), window_size=10 becomes 100, total=200 <= 200
    with imperial_unit_system:
        SimulationParams(
            models=[Fluid(), wall_1],
            time_stepping=Steady(max_steps=200),
            outputs=[
                fl.ForceOutput(
                    name="force_output",
                    models=[wall_1],
                    output_fields=["CL", "CD"],
                    moving_statistic=MovingStatistic(
                        method="mean", moving_window_size=10, start_step=100
                    ),
                )
            ],
        )


def test_check_moving_statistic_applicability_multiple_outputs():
    """
    Test moving_statistic with multiple outputs - captures ALL errors from different outputs.

    The validation function collects all errors from all invalid outputs and raises them together.
    This follows Pydantic's pattern of collecting errors from list items.
    """
    wall_1 = Wall(entities=Surface(name="fluid/wing"))
    uv_surface1 = UserVariable(
        name="uv_surface1", value=math.dot(solution.velocity, solution.CfVec)
    )

    # Multiple outputs with errors - ALL errors should be collected
    # All 4 outputs have invalid moving_statistic (500 + 600 = 1100 > 1000)
    with pytest.raises(pydantic.ValidationError) as exc_info:
        with imperial_unit_system:
            SimulationParams(
                models=[Fluid(), wall_1],
                time_stepping=Unsteady(steps=1000, step_size=1e-3),
                outputs=[
                    fl.ForceOutput(
                        name="force_output",
                        models=[wall_1],
                        output_fields=["CL", "CD"],
                        moving_statistic=MovingStatistic(
                            method="mean", moving_window_size=100, start_step=600
                        ),
                    ),
                    ProbeOutput(
                        name="probe_output",
                        probe_points=[Point(name="point_1", location=[1, 2, 3] * u.m)],
                        output_fields=["Cp"],
                        moving_statistic=MovingStatistic(
                            method="standard_deviation", moving_window_size=100, start_step=600
                        ),
                    ),
                    SurfaceIntegralOutput(
                        entities=Surface(name="fluid/wing"),
                        output_fields=[uv_surface1],
                        moving_statistic=MovingStatistic(
                            method="mean", moving_window_size=500, start_step=600
                        ),
                    ),
                    SurfaceProbeOutput(
                        name="surface_probe_output",
                        probe_points=[Point(name="point_1", location=[1, 2, 3] * u.m)],
                        output_fields=["Cp"],
                        target_surfaces=[Surface(name="fluid/wing")],
                        moving_statistic=MovingStatistic(
                            method="mean", moving_window_size=100, start_step=600
                        ),
                    ),
                ],
            )

    assert len(exc_info.value.errors()) == 1
    assert_validation_error_contains(
        exc_info.value,
        expected_loc=("outputs", 2, "moving_statistic"),
        expected_msg_contains="`moving_statistic`'s moving_window_size + start_step exceeds "
        "the total number of steps in the simulation.",
    )


def test_check_moving_statistic_applicability_no_moving_statistic():
    """Test that outputs without moving_statistic are not validated."""
    wall_1 = Wall(entities=Surface(name="fluid/wing"))

    # Should pass - no moving_statistic specified
    with imperial_unit_system:
        SimulationParams(
            models=[Fluid(), wall_1],
            time_stepping=Steady(max_steps=1000),
            outputs=[
                fl.ForceOutput(
                    name="force_output",
                    models=[wall_1],
                    output_fields=["CL", "CD"],
                ),
                ProbeOutput(
                    name="probe_output",
                    probe_points=[Point(name="point_1", location=[1, 2, 3] * u.m)],
                    output_fields=["Cp"],
                ),
            ],
        )


def test_check_moving_statistic_applicability_no_time_stepping():
    """Test that function returns early when no time_stepping is provided."""
    wall_1 = Wall(entities=Surface(name="fluid/wing"))

    # Should pass - no time_stepping specified
    with imperial_unit_system:
        SimulationParams(
            models=[Fluid(), wall_1],
            outputs=[
                fl.ForceOutput(
                    name="force_output",
                    models=[wall_1],
                    output_fields=["CL", "CD"],
                    moving_statistic=MovingStatistic(
                        method="mean", moving_window_size=100, start_step=200
                    ),
                )
            ],
        )


def test_duplicate_probe_names():

    # should have no error
    with imperial_unit_system:
        SimulationParams(
            outputs=[
                ProbeOutput(
                    name="probe_output_1",
                    probe_points=[Point(name="point_1", location=[1, 2, 3] * u.m)],
                    output_fields=["Cp"],
                ),
                ProbeOutput(
                    name="probe_output_2",
                    probe_points=[Point(name="point_1", location=[1, 2, 3] * u.m)],
                    output_fields=["velocity_x"],
                ),
            ],
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "`outputs`[1] ProbeOutput: Output name probe_output has already been used for a "
            "`ProbeOutput` or `SurfaceProbeOutput`. Output names must be unique among all probe outputs."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                outputs=[
                    ProbeOutput(
                        name="probe_output",
                        probe_points=[Point(name="point_1", location=[1, 2, 3] * u.m)],
                        output_fields=["Cp"],
                    ),
                    ProbeOutput(
                        name="probe_output",
                        probe_points=[Point(name="point_1", location=[1, 2, 3] * u.m)],
                        output_fields=["velocity_x"],
                    ),
                ],
            )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "`outputs`[1] SurfaceProbeOutput: Output name probe_output has already been used for a "
            "`ProbeOutput` or `SurfaceProbeOutput`. Output names must be unique among all probe outputs."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                outputs=[
                    ProbeOutput(
                        name="probe_output",
                        probe_points=[Point(name="point_1", location=[1, 2, 3] * u.m)],
                        output_fields=["pressure"],
                    ),
                    SurfaceProbeOutput(
                        name="probe_output",
                        probe_points=[Point(name="point_1", location=[1, 2, 3] * u.m)],
                        output_fields=["velocity_y"],
                        target_surfaces=[Surface(name="fluid/body")],
                    ),
                ],
            )


def test_duplicate_force_distribution_names():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "`outputs`[1] TimeAverageForceDistributionOutput: Output name test has already been used for a "
            "`ForceDistributionOutput`. Output names must be unique among all force distribution outputs."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                outputs=[
                    ForceDistributionOutput(
                        name="test",
                        distribution_direction=[1.0, 0.0, 0.0],
                    ),
                    TimeAverageForceDistributionOutput(
                        name="test",
                        distribution_direction=[0.0, 1.0, 0.0],
                    ),
                ],
            )


def test_time_averaged_force_distribution_output_requires_unsteady():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "`TimeAverageForceDistributionOutput` can only be used in unsteady simulations."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                models=[Fluid(turbulence_model_solver=NoneSolver())],
                outputs=[
                    TimeAverageForceDistributionOutput(
                        name="test",
                        distribution_direction=[1.0, 0.0, 0.0],
                        start_step=10,
                    ),
                ],
                time_stepping=Steady(),
            )


def test_duplicate_probe_entity_names(mock_validation_context):

    # should have no error
    with imperial_unit_system, mock_validation_context:
        SimulationParams(
            outputs=[
                ProbeOutput(
                    name="probe_output",
                    probe_points=[
                        Point(name="point_1", location=[1, 2, 3] * u.m),
                        Point(name="point_2", location=[1, 2, 3] * u.m),
                    ],
                    output_fields=["Cp"],
                ),
                ProbeOutput(
                    name="probe_output2",
                    probe_points=[Point(name="point_1", location=[1, 2, 3] * u.m)],
                    output_fields=["velocity_x"],
                ),
            ],
        )

    with mock_validation_context, pytest.raises(
        ValueError,
        match=re.escape(
            "In `outputs`[0] ProbeOutput: Entity name point_1 has already been used in the "
            "same `ProbeOutput`. Entity names must be unique."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                outputs=[
                    ProbeOutput(
                        name="probe_output_1",
                        probe_points=[
                            Point(name="point_1", location=[1, 2, 3] * u.m),
                            Point(name="point_1", location=[1, 2, 3] * u.m),
                        ],
                        output_fields=["Cp"],
                    ),
                    ProbeOutput(
                        name="probe_output_2",
                        probe_points=[Point(name="point_1", location=[1, 2, 3] * u.m)],
                        output_fields=["velocity_x"],
                    ),
                ],
            )

    with mock_validation_context, pytest.raises(
        ValueError,
        match=re.escape(
            "In `outputs`[0] SurfaceProbeOutput: Entity name point_1 has already been used in the "
            "same `SurfaceProbeOutput`. Entity names must be unique."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                outputs=[
                    SurfaceProbeOutput(
                        name="probe_output_1",
                        probe_points=[
                            Point(name="point_1", location=[1, 2, 3] * u.m),
                            Point(name="point_1", location=[1, 2, 3] * u.m),
                        ],
                        output_fields=["pressure"],
                        target_surfaces=[Surface(name="fluid/body")],
                    ),
                    SurfaceProbeOutput(
                        name="probe_output_2",
                        probe_points=[Point(name="point_1", location=[1, 2, 3] * u.m)],
                        output_fields=["velocity_y"],
                        target_surfaces=[Surface(name="fluid/body")],
                    ),
                ],
            )


def test_surface_integral_entity_types(mock_validation_context):
    uv_surface1 = UserVariable(
        name="uv_surface1", value=math.dot(solution.velocity, solution.CfVec)
    )
    surface = Surface(name="fluid/body")
    imported_surface = ImportedSurface(name="imported", file_name="imported.stl")
    with imperial_unit_system:
        SimulationParams(
            outputs=[
                SurfaceIntegralOutput(entities=surface, output_fields=[uv_surface1]),
                SurfaceIntegralOutput(
                    entities=imported_surface,
                    output_fields=[uv_surface1],
                ),
            ],
        )

    with mock_validation_context, pytest.raises(
        ValueError,
        match=re.escape(
            "Imported and simulation surfaces cannot be used together in the same SurfaceIntegralOutput."
            " Please assign them to separate outputs."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                outputs=[
                    SurfaceIntegralOutput(
                        entities=[surface, imported_surface], output_fields=[uv_surface1]
                    ),
                ],
            )


def test_imported_surface_output_fields_validation(mock_validation_context):
    """Test that imported surfaces only allow CommonFieldNames and Volume solver variables"""
    imported_surface = ImportedSurface(name="imported", file_name="imported.stl")
    surface = Surface(name="fluid/body")

    # Test 1: Surface-specific field name (not in CommonFieldNames) should fail with imported surface
    with mock_validation_context, pytest.raises(
        ValueError,
        match=re.escape(
            "Output field 'Cf' is not allowed for imported surfaces. "
            "Only non-Surface field names are allowed for string format output fields when using imported surfaces."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                outputs=[
                    SurfaceOutput(
                        entities=imported_surface,
                        output_fields=["Cf"],  # Cf is surface-specific, not in CommonFieldNames
                    )
                ],
            )

    # Test 2: UserVariable with Surface solver variables should fail with imported surface
    uv_surface = UserVariable(name="uv_surface", value=math.dot(solution.velocity, solution.CfVec))
    with mock_validation_context, pytest.raises(
        ValueError,
        match=re.escape(
            "Variable `uv_surface` cannot be used with imported surfaces "
            "since it contains Surface type solver variable(s): solution.CfVec. "
            "Only Volume type solver variables and 'solution.node_unit_normal' are allowed."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                outputs=[
                    SurfaceOutput(
                        entities=imported_surface,
                        output_fields=[uv_surface],
                    )
                ],
            )

    # Test 3: Multiple Surface solver variables in UserVariable should fail with imported surface
    uv_multiple_surface = UserVariable(
        name="uv_multiple",
        value=solution.node_forces_per_unit_area[0] * solution.Cp * solution.Cf,
    )
    with mock_validation_context, pytest.raises(
        ValueError,
        match=re.escape(
            "Variable `uv_multiple` cannot be used with imported surfaces "
            "since it contains Surface type solver variable(s): solution.Cf, solution.node_forces_per_unit_area. "
            "Only Volume type solver variables and 'solution.node_unit_normal' are allowed."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                outputs=[
                    SurfaceOutput(
                        entities=imported_surface,
                        output_fields=[uv_multiple_surface],
                    )
                ],
            )

    # Test 4: CommonFieldNames should work with imported surface
    with mock_validation_context, imperial_unit_system:
        SimulationParams(
            outputs=[
                SurfaceOutput(
                    entities=imported_surface,
                    output_fields=["Cp", "Mach"],  # These are CommonFieldNames
                )
            ],
        )

    # Test 5: UserVariable with only Volume solver variables should work with imported surface
    uv_volume = UserVariable(
        name="uv_volume", value=math.dot(solution.velocity, solution.vorticity)
    )
    with mock_validation_context, imperial_unit_system:
        SimulationParams(
            outputs=[
                SurfaceOutput(
                    entities=imported_surface,
                    output_fields=[uv_volume],
                )
            ],
        )
    # Test 5.5: UserVariable with node_unit_normal should work with imported surface
    uv_node_normal = UserVariable(
        name="uv_node_normal", value=math.dot(solution.velocity, solution.node_unit_normal)
    )
    with mock_validation_context, imperial_unit_system:
        SimulationParams(
            outputs=[
                SurfaceOutput(
                    entities=imported_surface,
                    output_fields=[uv_node_normal],
                )
            ],
        )

    # Test 6: Regular surfaces should not be affected - surface-specific fields should work
    with mock_validation_context, imperial_unit_system:
        SimulationParams(
            outputs=[
                SurfaceOutput(
                    entities=surface,
                    output_fields=[
                        "Cf"
                    ],  # Surface-specific fields should work for regular surfaces
                )
            ],
        )

    # Test 7: Mixed entities (imported + regular surfaces) should trigger validation
    with mock_validation_context, pytest.raises(
        ValueError,
        match=re.escape(
            "Output field 'Cf' is not allowed for imported surfaces. "
            "Only non-Surface field names are allowed for string format output fields when using imported surfaces."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                outputs=[
                    SurfaceOutput(
                        entities=[surface, imported_surface],
                        output_fields=["Cf"],
                    )
                ],
            )


def test_imported_surface_output_fields_validation_surface_integral(mock_validation_context):
    """Test that imported surfaces in SurfaceIntegralOutput only allow CommonFieldNames and Volume solver variables"""
    imported_surface = ImportedSurface(name="imported", file_name="imported.stl")
    surface = Surface(name="fluid/body")

    # Test 1: UserVariable with Surface solver variables should fail with imported surface
    uv_surface = UserVariable(name="uv_surface", value=math.dot(solution.velocity, solution.CfVec))
    with mock_validation_context, pytest.raises(
        ValueError,
        match=re.escape(
            "Variable `uv_surface` cannot be used with imported surfaces "
            "since it contains Surface type solver variable(s): solution.CfVec. "
            "Only Volume type solver variables and 'solution.node_unit_normal' are allowed."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                outputs=[
                    SurfaceIntegralOutput(
                        entities=imported_surface,
                        output_fields=[uv_surface],
                    )
                ],
            )

    # Test 2: UserVariable with only Volume solver variables should work with imported surface
    uv_volume = UserVariable(
        name="uv_volume", value=math.dot(solution.velocity, solution.vorticity)
    )
    with mock_validation_context, imperial_unit_system:
        SimulationParams(
            outputs=[
                SurfaceIntegralOutput(
                    entities=imported_surface,
                    output_fields=[uv_volume],
                )
            ],
        )

    # Test 2.5: UserVariable with node_unit_normal should work with imported surface (MassFluxProjected use case)
    uv_mass_flux_projected = UserVariable(
        name="MassFluxProjected",
        value=math.dot(solution.density * solution.velocity, solution.node_unit_normal),
    )
    with mock_validation_context, imperial_unit_system:
        SimulationParams(
            outputs=[
                SurfaceIntegralOutput(
                    entities=imported_surface,
                    output_fields=[uv_mass_flux_projected],
                )
            ],
        )

    # Test 3: Regular surfaces should not be affected - surface variables should work
    with mock_validation_context, imperial_unit_system:
        SimulationParams(
            outputs=[
                SurfaceIntegralOutput(
                    entities=surface,
                    output_fields=[uv_surface],
                )
            ],
        )


def test_output_frequency_settings_in_steady_simulation():
    volume_mesh = VolumeMeshV2.from_local_storage(
        mesh_id=None,
        local_storage_path=os.path.join(
            os.path.dirname(__file__), "..", "data", "vm_entity_provider"
        ),
    )
    with open(
        os.path.join(
            os.path.dirname(__file__), "..", "data", "vm_entity_provider", "simulation.json"
        ),
        "r",
    ) as fh:
        asset_cache_data = json.load(fh).pop("private_attribute_asset_cache")
    asset_cache = AssetCache.model_validate(asset_cache_data)
    with imperial_unit_system:
        params = SimulationParams(
            models=[Wall(name="wall", entities=volume_mesh["*"])],
            time_stepping=Steady(),
            outputs=[
                VolumeOutput(
                    output_fields=["Mach", "Cp"],
                    frequency=2,
                ),
                SurfaceOutput(
                    output_fields=["Cp"],
                    entities=volume_mesh["*"],
                    frequency_offset=10,
                ),
            ],
            private_attribute_asset_cache=asset_cache,
        )

    params_as_dict = params.model_dump(exclude_none=True, mode="json")
    params, errors, _ = validate_model(
        params_as_dict=params_as_dict,
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
        validation_level="All",
    )

    expected_errors = [
        {
            "loc": ("outputs", 0, "frequency"),
            "type": "value_error",
            "msg": "Value error, Output frequency cannot be specified in a steady simulation.",
            "ctx": {"relevant_for": ["Case"]},
        },
        {
            "loc": ("outputs", 1, "frequency_offset"),
            "type": "value_error",
            "msg": "Value error, Output frequency_offset cannot be specified in a steady simulation.",
            "ctx": {"relevant_for": ["Case"]},
        },
    ]


def test_force_output_with_wall_models():
    """Test ForceOutput with Wall models works correctly."""
    wall_1 = Wall(entities=Surface(name="fluid/wing1"))
    wall_2 = Wall(entities=Surface(name="fluid/wing2"))

    with imperial_unit_system:
        SimulationParams(
            models=[Fluid(), wall_1, wall_2],
            outputs=[
                fl.ForceOutput(
                    name="force_output",
                    models=[wall_1, wall_2],
                    output_fields=["CL", "CD", "CMx"],
                )
            ],
        )

    # Test with extended force coefficients (SkinFriction/Pressure)
    with imperial_unit_system:
        SimulationParams(
            models=[Fluid(), wall_1],
            outputs=[
                fl.ForceOutput(
                    name="force_output",
                    models=[wall_1],
                    output_fields=["CLSkinFriction", "CLPressure", "CDSkinFriction"],
                )
            ],
        )


def test_force_output_with_surface_and_volume_models(mock_validation_context):
    """Test ForceOutput with volume models (BETDisk, ActuatorDisk, PorousMedium)."""
    wall_1 = Wall(entities=Surface(name="fluid/wing"))
    with imperial_unit_system:
        fluid_model = Fluid()
        porous_zone = fl.Box.from_principal_axes(
            name="box",
            axes=[[0, 1, 0], [0, 0, 1]],
            center=[0, 0, 0] * fl.u.m,
            size=[0.2, 0.3, 2] * fl.u.m,
        )
        porous_medium = PorousMedium(
            entities=[porous_zone],
            darcy_coefficient=[1e6, 0, 0],
            forchheimer_coefficient=[1, 0, 0],
            volumetric_heat_source=0,
        )

    # Valid case: only basic force coefficients
    mock_validation_context.info.physics_model_dict = {
        fluid_model.private_attribute_id: fluid_model,
        wall_1.private_attribute_id: wall_1,
        porous_medium.private_attribute_id: porous_medium,
    }
    with imperial_unit_system, mock_validation_context:
        SimulationParams(
            models=[Fluid(), wall_1, porous_medium],
            outputs=[
                fl.ForceOutput(
                    name="force_output",
                    models=[wall_1, porous_medium],
                    output_fields=["CL", "CD", "CFx", "CFy", "CFz", "CMx", "CMy", "CMz"],
                )
            ],
        )

    mock_validation_context.info.physics_model_dict = {
        fluid_model.private_attribute_id: fluid_model,
        wall_1.private_attribute_id: wall_1,
        porous_medium.private_attribute_id: porous_medium,
    }
    with mock_validation_context, pytest.raises(
        ValueError,
        match=re.escape(
            "When ActuatorDisk/BETDisk/PorousMedium is specified, "
            "only CL, CD, CFx, CFy, CFz, CMx, CMy, CMz can be set as output_fields."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                models=[Fluid(), wall_1, porous_medium],
                outputs=[
                    fl.ForceOutput(
                        name="force_output",
                        models=[wall_1, porous_medium],
                        output_fields=["CL", "CLSkinFriction"],
                    )
                ],
            )


def test_force_output_duplicate_models():
    """Test that ForceOutput rejects duplicate models."""
    wall_1 = Wall(entities=Surface(name="fluid/wing"))

    with pytest.raises(
        ValueError,
        match=re.escape("Duplicate models are not allowed in the same `ForceOutput`."),
    ):
        with imperial_unit_system:
            SimulationParams(
                models=[Fluid(), wall_1],
                outputs=[
                    fl.ForceOutput(
                        name="force_output",
                        models=[wall_1, wall_1],
                        output_fields=["CL", "CD"],
                    )
                ],
            )


def test_force_output_nonexistent_model():
    """Test that ForceOutput rejects models not in SimulationParams' models list."""
    wall_1 = Wall(entities=Surface(name="fluid/wing1"))
    wall_2 = Wall(entities=Surface(name="fluid/wing2"))

    non_wall2_context = ParamsValidationInfo({}, [])
    non_wall2_context.physics_model_dict = {wall_1.private_attribute_id: wall_1.model_dump()}

    with ValidationContext(CASE, non_wall2_context), pytest.raises(
        ValueError,
        match=re.escape("The model does not exist in simulation params' models list."),
    ):
        with imperial_unit_system:
            SimulationParams(
                models=[Fluid(), wall_1],
                outputs=[
                    fl.ForceOutput(
                        name="force_output",
                        models=[wall_2.private_attribute_id],
                        output_fields=["CL", "CD"],
                    )
                ],
            )


def test_force_output_with_moving_statistic():
    """Test ForceOutput with moving statistics."""
    wall_1 = Wall(entities=Surface(name="fluid/wing"))

    with imperial_unit_system:
        SimulationParams(
            models=[Fluid(), wall_1],
            outputs=[
                fl.ForceOutput(
                    name="force_output",
                    models=[wall_1],
                    output_fields=["CL", "CD"],
                    moving_statistic=fl.MovingStatistic(
                        method="mean", moving_window_size=20, start_step=100
                    ),
                )
            ],
        )


def test_force_output_with_model_id():
    # [Frontend] Simulating loading a ForceOutput object with the id of models,
    # ensure the validation for models works
    with open("data/simulation_force_output_webui.json", "r") as fh:
        data = json.load(fh)

    _, errors, _ = validate_model(
        params_as_dict=data, validated_by=ValidationCalledBy.LOCAL, root_item_type="VolumeMesh"
    )
    # Expected errors:
    # - outputs[3,4,5] have validation errors in their models field
    # - Since outputs field has validation errors, the output_dict is not populated
    # - Therefore all stopping_criteria that reference outputs by ID fail with a clear error message
    expected_errors = [
        {
            "type": "value_error",
            "loc": ("outputs", 3, "models"),
            "msg": "Value error, Duplicate models are not allowed in the same `ForceOutput`.",
            "ctx": {"relevant_for": ["Case"]},
        },
        {
            "type": "value_error",
            "loc": ("outputs", 4, "models"),
            "msg": "Value error, When ActuatorDisk/BETDisk/PorousMedium is specified, "
            "only CL, CD, CFx, CFy, CFz, CMx, CMy, CMz can be set as output_fields.",
            "ctx": {"relevant_for": ["Case"]},
        },
        {
            "type": "value_error",
            "loc": ("outputs", 5, "models"),
            "msg": "Value error, The model does not exist in simulation params' models list.",
            "ctx": {"relevant_for": ["Case"]},
        },
    ]

    assert len(errors) == len(expected_errors)
    for err, exp_err in zip(errors, expected_errors):
        assert err["loc"] == exp_err["loc"]
        assert err["type"] == exp_err["type"]
        assert err["ctx"]["relevant_for"] == exp_err["ctx"]["relevant_for"]
        assert err["msg"] == exp_err["msg"]


def test_force_distribution_output_entities_validation():
    """Test ForceDistributionOutput entities validation."""

    # Test 1: Valid case - ForceDistributionOutput without entities (default all walls)
    with imperial_unit_system:
        SimulationParams(
            outputs=[
                ForceDistributionOutput(
                    name="test_default",
                    distribution_direction=[1.0, 0.0, 0.0],
                ),
            ],
        )

    # Test 2: Valid case - ForceDistributionOutput with surface entities
    with imperial_unit_system:
        SimulationParams(
            outputs=[
                ForceDistributionOutput(
                    name="test_with_surfaces",
                    distribution_direction=[1.0, 0.0, 0.0],
                    entities=[Surface(name="fluid/wing")],
                ),
            ],
        )

    # Test 3: Valid case - TimeAverageForceDistributionOutput with entities
    with imperial_unit_system:
        SimulationParams(
            outputs=[
                TimeAverageForceDistributionOutput(
                    name="test_time_avg",
                    distribution_direction=[0.0, 1.0, 0.0],
                    entities=[Surface(name="fluid/body")],
                    start_step=10,
                ),
            ],
            time_stepping=Unsteady(steps=100, step_size=1e-3),
        )

    # Test 4: Valid case - ForceDistributionOutput with multiple surfaces
    with imperial_unit_system:
        SimulationParams(
            outputs=[
                ForceDistributionOutput(
                    name="test_multiple_surfaces",
                    distribution_direction=[0.0, 0.0, 1.0],
                    entities=[
                        Surface(name="fluid/wing"),
                        Surface(name="fluid/fuselage"),
                    ],
                ),
            ],
        )

    # Test 5: Valid case - ForceDistributionOutput with custom number_of_segments
    with imperial_unit_system:
        SimulationParams(
            outputs=[
                ForceDistributionOutput(
                    name="test_custom_segments",
                    distribution_direction=[1.0, 0.0, 0.0],
                    entities=[Surface(name="fluid/wing")],
                    number_of_segments=500,
                ),
            ],
        )


def test_force_distribution_output_requires_wall_bc(mock_validation_context):
    """Test that ForceDistributionOutput validates surfaces have Wall BC."""
    from flow360.component.simulation.models.surface_models import Freestream, SlipWall

    wing_surface = Surface(name="fluid/wing")
    freestream_surface = Surface(name="fluid/farfield")

    # Test: Valid case - surface with Wall BC
    with mock_validation_context, imperial_unit_system:
        SimulationParams(
            models=[
                Fluid(),
                Wall(entities=[wing_surface]),
                Freestream(entities=[freestream_surface]),
            ],
            outputs=[
                ForceDistributionOutput(
                    name="test_valid",
                    distribution_direction=[1.0, 0.0, 0.0],
                    entities=[wing_surface],
                ),
            ],
        )

    # Test: Invalid case - surface without Wall BC (has Freestream BC)
    with mock_validation_context, pytest.raises(
        ValueError,
        match=re.escape("The following surfaces do not have Wall boundary conditions assigned"),
    ):
        with imperial_unit_system:
            SimulationParams(
                models=[
                    Fluid(),
                    Wall(entities=[wing_surface]),
                    Freestream(entities=[freestream_surface]),
                ],
                outputs=[
                    ForceDistributionOutput(
                        name="test_invalid",
                        distribution_direction=[1.0, 0.0, 0.0],
                        entities=[freestream_surface],  # This has Freestream BC, not Wall
                    ),
                ],
            )

    # Test: Invalid case - surface with SlipWall BC (not a no-slip Wall)
    slipwall_surface = Surface(name="fluid/symmetry")
    with mock_validation_context, pytest.raises(
        ValueError,
        match=re.escape("The following surfaces do not have Wall boundary conditions assigned"),
    ):
        with imperial_unit_system:
            SimulationParams(
                models=[
                    Fluid(),
                    Wall(entities=[wing_surface]),
                    SlipWall(entities=[slipwall_surface]),
                ],
                outputs=[
                    ForceDistributionOutput(
                        name="test_slipwall",
                        distribution_direction=[1.0, 0.0, 0.0],
                        entities=[slipwall_surface],  # SlipWall is not Wall
                    ),
                ],
            )


def test_surface_output_write_single_file_validator():
    with pytest.raises(
        ValueError,
        match=re.escape("write_single_file is only supported for Tecplot output format."),
    ):
        SurfaceOutput(
            write_single_file=True,
            entities=[Surface(name="noSlipWall")],
            output_fields=["Cp"],
            output_format="paraview",
        )

    SurfaceOutput(
        write_single_file=True,
        entities=[Surface(name="noSlipWall")],
        output_fields=["Cp"],
        output_format="tecplot",
    )

    SurfaceOutput(
        write_single_file=True,
        entities=[Surface(name="noSlipWall")],
        output_fields=["Cp"],
        output_format="both",
    )
