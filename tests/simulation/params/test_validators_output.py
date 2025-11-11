import json
import os
import re

import pydantic as pd
import pytest

import flow360 as fl
import flow360.component.simulation.units as u
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
    ProbeOutput,
    SurfaceOutput,
    SurfaceProbeOutput,
    TimeAverageSurfaceOutput,
    VolumeOutput,
)
from flow360.component.simulation.primitives import Surface
from flow360.component.simulation.services import (
    ValidationCalledBy,
    clear_context,
    validate_model,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import Steady, Unsteady
from flow360.component.simulation.unit_system import imperial_unit_system
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


def test_turbulence_enabled_output_fields():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "In `outputs`[0] IsosurfaceOutput:, kOmega is not a valid output field when using turbulence model: None."
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
            "In `outputs`[0] IsosurfaceOutput:, nuHat is not a valid iso field when using turbulence model: kOmegaSST."
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
            "In `outputs`[0] VolumeOutput:, kOmega is not a valid output field when using turbulence model: SpalartAllmaras."
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
            "In `outputs`[0] IsosurfaceOutput:, solutionTransition is not a valid output field when transition model is not used."
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
            "In `outputs`[0] SurfaceProbeOutput:, residualTransition is not a valid output field when transition model is not used."
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
            "In `outputs`[0] VolumeOutput:, linearResidualTransition is not a valid output field when transition model is not used."
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


def test_output_frequecy_settings_in_steady_simulation():

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
    print("haha", params_as_dict)

    params, errors, _ = validate_model(
        params_as_dict=params_as_dict,
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
        validation_level="All",
    )

    print("jaja", errors)

    expected_errors = [
        {
            "loc": ("outputs", 0, "VolumeOutput"),
            "type": "value_error",
            "msg": "Value error, Output frequency cannot be specified in a steady simulation.",
            "ctx": {"relevant_for": ["Case"]},
        },
        {
            "loc": ("outputs", 1, "SurfaceOutput"),
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
