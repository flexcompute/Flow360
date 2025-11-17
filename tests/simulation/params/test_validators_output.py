import json
import os
import re

import pytest

import flow360 as fl
import flow360.component.simulation.units as u
from flow360.component.simulation.entity_info import VolumeMeshEntityInfo
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.models.solver_numerics import (
    KOmegaSST,
    NoneSolver,
    SpalartAllmaras,
)
from flow360.component.simulation.models.surface_models import Wall
from flow360.component.simulation.models.volume_models import Fluid
from flow360.component.simulation.outputs.output_entities import Point
from flow360.component.simulation.outputs.outputs import (
    AeroAcousticOutput,
    Isosurface,
    IsosurfaceOutput,
    MovingStatistic,
    ProbeOutput,
    SurfaceIntegralOutput,
    SurfaceOutput,
    SurfaceProbeOutput,
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
from flow360.component.volume_mesh import VolumeMeshV2


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


def test_duplicate_surface_usage():
    my_var = UserVariable(name="my_var", value=solution.node_forces_per_unit_area[1])
    with pytest.raises(
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

    with pytest.raises(
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


def test_moving_statitic_validator():
    wall_1 = Surface(name="wall_1", private_attribute_is_interface=False)
    asset_cache = AssetCache(
        project_length_unit="m",
        project_entity_info=VolumeMeshEntityInfo(boundaries=[wall_1]),
    )

    with SI_unit_system:
        monitored_variable = UserVariable(
            name="Helicity_MONITOR",
            value=math.dot(solution.velocity, solution.vorticity),
        )
        params = SimulationParams(
            time_stepping=Steady(max_steps=5000),
            models=[Fluid(), Wall(entities=wall_1)],
            outputs=[
                ProbeOutput(
                    name="point_legacy2",
                    output_fields=["Mach", monitored_variable],
                    probe_points=Point(name="Point1", location=(-0.026642, 0.56614, 0) * u.m),
                    moving_statistic=MovingStatistic(method="std", moving_window_size=15),
                )
            ],
            private_attribute_asset_cache=asset_cache,
        )

    params, errors, _ = validate_model(
        validated_by=ValidationCalledBy.LOCAL,
        params_as_dict=params.model_dump(mode="json"),
        root_item_type="VolumeMesh",
        validation_level="Case",
    )
    assert len(errors) == 1
    assert (
        errors[0]["msg"] == "Value error, For steady simulation, "
        "the number of steps should be a multiple of 10."
    )

    with SI_unit_system:
        monitored_variable = UserVariable(
            name="Helicity_MONITOR",
            value=math.dot(solution.velocity, solution.vorticity),
        )
        params = SimulationParams(
            time_stepping=Steady(max_steps=5000),
            models=[Fluid(), Wall(entities=wall_1)],
            outputs=[
                ProbeOutput(
                    name="point_legacy2",
                    output_fields=["Mach", monitored_variable],
                    probe_points=Point(name="Point1", location=(-0.026642, 0.56614, 0) * u.m),
                    moving_statistic=MovingStatistic(method="std", moving_window_size=20),
                )
            ],
            private_attribute_asset_cache=asset_cache,
        )

    _, errors, _ = validate_model(
        validated_by=ValidationCalledBy.LOCAL,
        params_as_dict=params.model_dump(mode="json"),
        root_item_type="VolumeMesh",
        validation_level="Case",
    )
    assert errors is None


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


def test_surface_integral_entity_types():
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

    with pytest.raises(
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
    assert len(errors) == len(expected_errors)
    for err, exp_err in zip(errors, expected_errors):
        assert err["loc"] == exp_err["loc"]
        assert err["type"] == exp_err["type"]
        assert err["ctx"]["relevant_for"] == exp_err["ctx"]["relevant_for"]
        assert err["msg"] == exp_err["msg"]
