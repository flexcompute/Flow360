import json

import numpy as np
import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.draft_context.coordinate_system_manager import (
    CoordinateSystemManager,
)
from flow360.component.simulation.entity_operation import CoordinateSystem
from flow360.component.simulation.framework.entity_registry import EntityRegistry
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.framework.updater_utils import compare_values
from flow360.component.simulation.models.material import Water
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
    LiquidOperatingCondition,
)
from flow360.component.simulation.outputs.output_entities import (
    Point,
    PointArray,
    PointArray2D,
)
from flow360.component.simulation.outputs.outputs import (
    AeroAcousticOutput,
    ForceDistributionOutput,
    Isosurface,
    IsosurfaceOutput,
    Observer,
    ProbeOutput,
    Slice,
    SliceOutput,
    StreamlineOutput,
    SurfaceIntegralOutput,
    SurfaceOutput,
    SurfaceProbeOutput,
    SurfaceSliceOutput,
    TimeAverageForceDistributionOutput,
    TimeAverageIsosurfaceOutput,
    TimeAverageProbeOutput,
    TimeAverageSurfaceOutput,
    TimeAverageSurfaceProbeOutput,
    TimeAverageVolumeOutput,
    UserDefinedField,
    VolumeOutput,
)
from flow360.component.simulation.primitives import ImportedSurface, Surface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import Unsteady
from flow360.component.simulation.translator.solver_translator import (
    get_solver_json,
    translate_output,
)
from flow360.component.simulation.unit_system import SI_unit_system
from flow360.component.simulation.user_code.core.types import UserVariable
from flow360.component.simulation.user_code.variables import solution


@pytest.fixture()
def vel_in_km_per_hr():
    return solution.velocity.in_units(new_name="velocity_in_km_per_hr", new_unit=u.km / u.hr)


@pytest.fixture()
def volume_output_config(vel_in_km_per_hr):
    return (
        VolumeOutput(
            frequency=1,
            frequency_offset=2,
            output_format="both",
            output_fields=[
                "primitiveVars",
                "betMetrics",
                "qcriterion",
                "velocity",
                "vorticity",
                vel_in_km_per_hr,
            ],
        ),
        {
            "animationFrequency": 1,
            "animationFrequencyOffset": 2,
            "animationFrequencyTimeAverage": -1,
            "animationFrequencyTimeAverageOffset": 0,
            "outputFields": [
                "betMetrics",
                "primitiveVars",
                "qcriterion",
                "velocity",
                "velocity_in_km_per_hr",
                "velocity_magnitude",
                "vorticity",
                "vorticityMagnitude",
            ],
            "outputFormat": "paraview,tecplot",
            "startAverageIntegrationStep": -1,
        },
    )


@pytest.fixture()
def avg_volume_output_config(vel_in_km_per_hr):
    return (
        TimeAverageVolumeOutput(
            frequency=11,
            frequency_offset=12,
            output_format="both",
            output_fields=[
                "primitiveVars",
                "betMetrics",
                "qcriterion",
                "velocity",
                vel_in_km_per_hr,
            ],
            start_step=1,
        ),
        {
            "animationFrequency": -1,
            "animationFrequencyOffset": 0,
            "animationFrequencyTimeAverage": 11,
            "animationFrequencyTimeAverageOffset": 12,
            "outputFields": [
                "betMetrics",
                "primitiveVars",
                "qcriterion",
                "velocity",
                "velocity_in_km_per_hr",
                "velocity_magnitude",
            ],
            "outputFormat": "paraview,tecplot",
            "startAverageIntegrationStep": 1,
        },
    )


def test_volume_output(volume_output_config, avg_volume_output_config):

    ##:: volumeOutput only
    with SI_unit_system:
        param = SimulationParams(outputs=[volume_output_config[0]])
    translated = {"boundaries": {}}
    translated = translate_output(param, translated)
    assert compare_values(volume_output_config[1], translated["volumeOutput"])

    ##:: timeAverageVolumeOutput only
    with SI_unit_system:
        param = SimulationParams(
            time_stepping=Unsteady(step_size=0.1 * u.s, steps=10),
            outputs=[avg_volume_output_config[0]],
        )
    translated = {"boundaries": {}}
    translated = translate_output(param, translated)
    assert compare_values(avg_volume_output_config[1], translated["timeAverageVolumeOutput"])

    ##:: timeAverageVolumeOutput and volumeOutput
    with SI_unit_system:
        param = SimulationParams(
            time_stepping=Unsteady(step_size=0.1 * u.s, steps=10),
            outputs=[volume_output_config[0], avg_volume_output_config[0]],
        )
    translated = {"boundaries": {}}
    translated = translate_output(param, translated)
    ref = {
        "volumeOutput": {
            "animationFrequency": 1,
            "animationFrequencyOffset": 2,
            "animationFrequencyTimeAverage": -1,
            "animationFrequencyTimeAverageOffset": 0,
            "outputFields": [
                "betMetrics",
                "primitiveVars",
                "qcriterion",
                "velocity",
                "velocity_in_km_per_hr",
                "velocity_magnitude",
                "vorticity",
                "vorticityMagnitude",
            ],
            "outputFormat": "paraview,tecplot",
            "startAverageIntegrationStep": -1,
        },
        "timeAverageVolumeOutput": {
            "animationFrequency": -1,
            "animationFrequencyOffset": 0,
            "animationFrequencyTimeAverage": 11,
            "animationFrequencyTimeAverageOffset": 12,
            "outputFields": [
                "betMetrics",
                "primitiveVars",
                "qcriterion",
                "velocity",
                "velocity_in_km_per_hr",
                "velocity_magnitude",
            ],
            "outputFormat": "paraview,tecplot",
            "startAverageIntegrationStep": 1,
        },
    }
    assert compare_values(ref["volumeOutput"], translated["volumeOutput"])
    assert compare_values(ref["timeAverageVolumeOutput"], translated["timeAverageVolumeOutput"])


@pytest.fixture()
def surface_output_config(vel_in_km_per_hr):
    return (
        [
            SurfaceOutput(  # Local
                entities=[Surface(name="surface1"), Surface(name="surface2")],
                output_fields=["Cp", vel_in_km_per_hr],
                output_format="tecplot",
                frequency=123,
                frequency_offset=321,
            ),
            SurfaceOutput(  # Local
                entities=[Surface(name="surface11"), Surface(name="surface22")],
                frequency=123,
                frequency_offset=321,
                output_fields=["T", "velocity", "vorticity", vel_in_km_per_hr],
                output_format="tecplot",
            ),
        ],
        {
            "animationFrequency": 123,
            "animationFrequencyOffset": 321,
            "animationFrequencyTimeAverage": -1,
            "animationFrequencyTimeAverageOffset": 0,
            "outputFields": [],
            "outputFormat": "tecplot",
            "startAverageIntegrationStep": -1,
            "surfaces": {
                "surface1": {"outputFields": ["Cp", "velocity_in_km_per_hr"]},
                "surface11": {
                    "outputFields": [
                        "T",
                        "velocity",
                        "velocity_in_km_per_hr",
                        "velocity_magnitude",
                        "vorticity",
                        "vorticityMagnitude",
                    ]
                },
                "surface2": {"outputFields": ["Cp", "velocity_in_km_per_hr"]},
                "surface22": {
                    "outputFields": [
                        "T",
                        "velocity",
                        "velocity_in_km_per_hr",
                        "velocity_magnitude",
                        "vorticity",
                        "vorticityMagnitude",
                    ]
                },
            },
            "writeSingleFile": False,
        },
    )


@pytest.fixture()
def avg_surface_output_config(vel_in_km_per_hr):
    return [
        TimeAverageSurfaceOutput(  # Local
            frequency=111,
            frequency_offset=222,
            output_format="paraview",
            entities=[Surface(name="surface1"), Surface(name="surface2")],
            output_fields=["Cf", vel_in_km_per_hr],
        ),
        TimeAverageSurfaceOutput(  # Local
            entities=[Surface(name="surface3")],
            output_fields=["primitiveVars", vel_in_km_per_hr],
        ),
    ]


def test_surface_output(
    surface_output_config,
    avg_surface_output_config,
):
    ##:: surfaceOutput
    with SI_unit_system:
        param = SimulationParams(outputs=surface_output_config[0])
    translated = {"boundaries": {}}
    translated = translate_output(param, translated)
    assert compare_values(surface_output_config[1], translated["surfaceOutput"])

    ##:: timeAverageSurfaceOutput and surfaceOutput
    with SI_unit_system:
        param = SimulationParams(
            time_stepping=Unsteady(step_size=0.1 * u.s, steps=10),
            outputs=surface_output_config[0] + avg_surface_output_config,
        )
    translated = {"boundaries": {}}
    translated = translate_output(param, translated)
    ref = {
        "surfaceOutput": {
            "animationFrequency": 123,
            "animationFrequencyOffset": 321,
            "animationFrequencyTimeAverage": -1,
            "animationFrequencyTimeAverageOffset": 0,
            "outputFields": [],
            "outputFormat": "tecplot",
            "startAverageIntegrationStep": -1,
            "surfaces": {
                "surface1": {"outputFields": ["Cp", "velocity_in_km_per_hr"]},
                "surface11": {
                    "outputFields": [
                        "T",
                        "velocity",
                        "velocity_in_km_per_hr",
                        "velocity_magnitude",
                        "vorticity",
                        "vorticityMagnitude",
                    ]
                },
                "surface2": {"outputFields": ["Cp", "velocity_in_km_per_hr"]},
                "surface22": {
                    "outputFields": [
                        "T",
                        "velocity",
                        "velocity_in_km_per_hr",
                        "velocity_magnitude",
                        "vorticity",
                        "vorticityMagnitude",
                    ]
                },
            },
            "writeSingleFile": False,
        },
        "timeAverageSurfaceOutput": {
            "animationFrequency": -1,
            "animationFrequencyOffset": 0,
            "animationFrequencyTimeAverage": 111,
            "animationFrequencyTimeAverageOffset": 222,
            "outputFields": [],
            "outputFormat": "paraview",
            "startAverageIntegrationStep": -1,
            "surfaces": {
                "surface1": {"outputFields": ["Cf", "velocity_in_km_per_hr"]},
                "surface3": {
                    "outputFields": [
                        "primitiveVars",
                        "velocity_in_km_per_hr",
                    ]
                },
                "surface2": {"outputFields": ["Cf", "velocity_in_km_per_hr"]},
            },
            "writeSingleFile": False,
        },
    }
    assert compare_values(ref["surfaceOutput"], translated["surfaceOutput"])
    assert compare_values(ref["timeAverageSurfaceOutput"], translated["timeAverageSurfaceOutput"])


@pytest.fixture()
def slice_output_config(vel_in_km_per_hr):
    return (
        [
            SliceOutput(  # Local
                entities=[
                    Slice(
                        name="slice10",
                        normal=(0, 2, 0),
                        origin=(0.02, 0.03, 0.04) * u.m,
                    ),
                    Slice(
                        name="slice20",
                        normal=(3, 4, 0),
                        origin=(0.12, 0.13, 0.14) * u.m,
                    ),
                ],
                output_fields=[
                    "Cp",
                    "velocity",
                    "vorticity",
                    "vorticityMagnitude",
                    vel_in_km_per_hr,
                ],
                frequency=33,
                frequency_offset=22,
                output_format="tecplot",
            ),
            SliceOutput(  # Local
                entities=[
                    Slice(
                        name="slice01",
                        normal=(10, 0, 0),
                        origin=(10.02, 10.03, 10.04) * u.m,
                    ),
                    Slice(
                        name="slice02",
                        normal=(30, 0, 40),
                        origin=(6.12, 6.13, 6.14) * u.m,
                    ),
                ],
                frequency=33,
                frequency_offset=22,
                output_format="tecplot",
                output_fields=["T", "primitiveVars", vel_in_km_per_hr],
            ),
        ],
        {
            "animationFrequency": 33,
            "animationFrequencyOffset": 22,
            "animationFrequencyTimeAverage": -1,
            "animationFrequencyTimeAverageOffset": 0,
            "startAverageIntegrationStep": -1,
            "outputFields": [],
            "outputFormat": "tecplot",
            "slices": {
                "slice01": {
                    "outputFields": ["T", "primitiveVars", "velocity_in_km_per_hr"],
                    "sliceNormal": [1.0, 0.0, 0.0],
                    "sliceOrigin": [10.02, 10.03, 10.04],
                },
                "slice02": {
                    "outputFields": ["T", "primitiveVars", "velocity_in_km_per_hr"],
                    "sliceNormal": [0.6, 0.0, 0.8],
                    "sliceOrigin": [6.12, 6.13, 6.14],
                },
                "slice10": {
                    "outputFields": [
                        "Cp",
                        "velocity",
                        "velocity_in_km_per_hr",
                        "velocity_magnitude",
                        "vorticity",
                        "vorticityMagnitude",
                    ],
                    "sliceNormal": [0.0, 1.0, 0.0],
                    "sliceOrigin": [0.02, 0.03, 0.04],
                },
                "slice20": {
                    "outputFields": [
                        "Cp",
                        "velocity",
                        "velocity_in_km_per_hr",
                        "velocity_magnitude",
                        "vorticity",
                        "vorticityMagnitude",
                    ],
                    "sliceNormal": [0.6, 0.8, 0.0],
                    "sliceOrigin": [0.12, 0.13, 0.14],
                },
            },
        },
    )


def test_slice_output(
    slice_output_config,
):
    ##:: sliceOutput with NO global settings
    with SI_unit_system:
        param = SimulationParams(outputs=slice_output_config[0])
    param = param._preprocess(1.0 * u.m, exclude=["models"])
    translated = {"boundaries": {}}
    translated = translate_output(param, translated)

    assert compare_values(slice_output_config[1], translated["sliceOutput"])


@pytest.fixture()
def isosurface_output_config(vel_in_km_per_hr):
    return (
        [
            IsosurfaceOutput(  # Local
                entities=[
                    Isosurface(
                        name="isosurface 10",
                        iso_value=0.0001,
                        field="T",
                    ),
                    Isosurface(
                        name="isosurface 14",
                        iso_value=20.431,
                        field="qcriterion",
                    ),
                    Isosurface(
                        name="isosurface 15",
                        iso_value=0.1,
                        field="velocity_x",
                    ),
                    Isosurface(
                        name="isosurface 16",
                        iso_value=0.2,
                        field="vorticity_z",
                    ),
                ],
                output_fields=["Cp", vel_in_km_per_hr],
                frequency=332,
                frequency_offset=222,
                output_format="paraview",
            ),
            IsosurfaceOutput(  # Local
                entities=[
                    Isosurface(
                        name="isosurface 01",
                        iso_value=0.0001,
                        field="nuHat",
                    ),
                    Isosurface(
                        name="isosurface 02",
                        iso_value=1e4,
                        field="qcriterion",
                    ),
                ],
                frequency=332,
                frequency_offset=222,
                output_format="paraview",
                output_fields=["T", "primitiveVars", vel_in_km_per_hr],
            ),
        ],
        {
            "animationFrequency": 332,
            "animationFrequencyOffset": 222,
            "isoSurfaces": {
                "isosurface 01": {
                    "outputFields": ["T", "primitiveVars", "velocity_in_km_per_hr"],
                    "surfaceField": "nuHat",
                    "surfaceFieldMagnitude": 0.0001,
                },
                "isosurface 02": {
                    "outputFields": ["T", "primitiveVars", "velocity_in_km_per_hr"],
                    "surfaceField": "qcriterion",
                    "surfaceFieldMagnitude": 10000.0,
                },
                "isosurface 10": {
                    "outputFields": ["Cp", "velocity_in_km_per_hr"],
                    "surfaceField": "T",
                    "surfaceFieldMagnitude": 0.0001,
                },
                "isosurface 14": {
                    "outputFields": ["Cp", "velocity_in_km_per_hr"],
                    "surfaceField": "qcriterion",
                    "surfaceFieldMagnitude": 20.431,
                },
                "isosurface 15": {
                    "outputFields": ["Cp", "velocity_in_km_per_hr"],
                    "surfaceField": "velocity_x",
                    "surfaceFieldMagnitude": 0.1,
                },
                "isosurface 16": {
                    "outputFields": ["Cp", "velocity_in_km_per_hr"],
                    "surfaceField": "vorticity_z",
                    "surfaceFieldMagnitude": 0.2,
                },
            },
            "outputFields": [],
            "outputFormat": "paraview",
        },
    )


@pytest.fixture()
def time_average_isosurface_output_config():
    return (
        [
            TimeAverageIsosurfaceOutput(
                entities=[
                    Isosurface(
                        name="isosurface 10",
                        iso_value=0.0001,
                        field="T",
                    ),
                    Isosurface(
                        name="isosurface 14",
                        iso_value=20.431,
                        field="qcriterion",
                    ),
                    Isosurface(
                        name="isosurface 15",
                        iso_value=0.1,
                        field="velocity_x",
                    ),
                    Isosurface(
                        name="isosurface 16",
                        iso_value=0.2,
                        field="vorticity_z",
                    ),
                ],
                output_fields=["Cp"],
                frequency=332,
                frequency_offset=222,
                output_format="paraview",
            ),
            TimeAverageIsosurfaceOutput(
                entities=[
                    Isosurface(
                        name="isosurface 01",
                        iso_value=0.0001,
                        field="nuHat",
                    ),
                    Isosurface(
                        name="isosurface 02",
                        iso_value=1e4,
                        field="qcriterion",
                    ),
                ],
                frequency=332,
                frequency_offset=222,
                output_format="paraview",
                output_fields=["T", "primitiveVars"],
            ),
        ],
        {
            "animationFrequencyTimeAverage": 332,
            "animationFrequencyTimeAverageOffset": 222,
            "startAverageIntegrationStep": -1,
            "isoSurfaces": {
                "isosurface 01": {
                    "outputFields": ["T", "primitiveVars"],
                    "surfaceField": "nuHat",
                    "surfaceFieldMagnitude": 0.0001,
                },
                "isosurface 02": {
                    "outputFields": ["T", "primitiveVars"],
                    "surfaceField": "qcriterion",
                    "surfaceFieldMagnitude": 10000.0,
                },
                "isosurface 10": {
                    "outputFields": ["Cp"],
                    "surfaceField": "T",
                    "surfaceFieldMagnitude": 0.0001,
                },
                "isosurface 14": {
                    "outputFields": ["Cp"],
                    "surfaceField": "qcriterion",
                    "surfaceFieldMagnitude": 20.431,
                },
                "isosurface 15": {
                    "outputFields": ["Cp"],
                    "surfaceField": "velocity_x",
                    "surfaceFieldMagnitude": 0.1,
                },
                "isosurface 16": {
                    "outputFields": ["Cp"],
                    "surfaceField": "vorticity_z",
                    "surfaceFieldMagnitude": 0.2,
                },
            },
            "outputFields": [],
            "outputFormat": "paraview",
        },
    )


def test_isosurface_output(
    isosurface_output_config,
):
    ##:: isoSurface with NO global settings
    with SI_unit_system:
        param = SimulationParams(outputs=isosurface_output_config[0])
    translated = {"boundaries": {}}
    translated = translate_output(param, translated)
    assert compare_values(isosurface_output_config[1], translated["isoSurfaceOutput"])


def test_time_average_isosurface_output(
    time_average_isosurface_output_config,
):
    with SI_unit_system:
        param = SimulationParams(
            time_stepping=Unsteady(step_size=0.1 * u.s, steps=10),
            outputs=time_average_isosurface_output_config[0],
        )
    translated = {"boundaries": {}}
    translated = translate_output(param, translated)
    assert compare_values(
        time_average_isosurface_output_config[1], translated["timeAverageIsoSurfaceOutput"]
    )


@pytest.fixture()
def probe_output_config(vel_in_km_per_hr):
    return (
        [
            ProbeOutput(  # Local
                name="prb 10",
                entities=[
                    Point(
                        name="124",
                        location=[1, 1.02, 0.03] * u.cm,
                    ),
                    Point(
                        name="asdfg",
                        location=[0.0001, 0.02, 0.03] * u.m,
                    ),
                ],
                output_fields=["primitiveVars", "Cp", vel_in_km_per_hr],
            ),
            ProbeOutput(  # Local
                name="prb 12",
                entities=[
                    Point(
                        name="asnbgoujba",
                        location=[10, 10.02, 10.03] * u.cm,
                    ),
                ],
                output_fields=["primitiveVars", "Cp", vel_in_km_per_hr],
            ),
            TimeAverageProbeOutput(  # Local
                name="prb average",
                entities=[
                    Point(
                        name="a",
                        location=[10, 10.02, 10.03] * u.cm,
                    ),
                ],
                output_fields=["primitiveVars", "Cp", "T", vel_in_km_per_hr],
                frequency=10,
            ),
        ],
        {
            "outputFields": [],
            "monitors": {
                "prb 10": {
                    "monitorLocations": {
                        "124": [0.01, 0.0102, 0.0003],
                        "asdfg": [0.0001, 0.02, 0.03],
                    },
                    "type": "probe",
                    "outputFields": ["Cp", "primitiveVars", "velocity_in_km_per_hr"],
                    "computeTimeAverages": False,
                    "animationFrequency": 1,
                    "animationFrequencyOffset": 0,
                },
                "prb 12": {
                    "monitorLocations": {"asnbgoujba": [0.1, 0.1002, 0.1003]},
                    "type": "probe",
                    "outputFields": ["Cp", "primitiveVars", "velocity_in_km_per_hr"],
                    "computeTimeAverages": False,
                    "animationFrequency": 1,
                    "animationFrequencyOffset": 0,
                },
                "prb average": {
                    "monitorLocations": {"a": [0.1, 0.1002, 0.1003]},
                    "type": "probe",
                    "outputFields": ["Cp", "T", "primitiveVars", "velocity_in_km_per_hr"],
                    "computeTimeAverages": True,
                    "animationFrequency": 1,
                    "animationFrequencyOffset": 0,
                    "animationFrequencyTimeAverage": 10,
                    "animationFrequencyTimeAverageOffset": 0,
                    "startAverageIntegrationStep": -1,
                },
            },
        },
    )


@pytest.fixture()
def probe_output_with_point_array(vel_in_km_per_hr):
    return (
        [
            ProbeOutput(
                name="prb line",
                entities=[
                    PointArray(
                        name="Line 1",
                        start=[0.1, 0.2, 0.3] * u.m,
                        end=[1.1, 1.2, 1.3] * u.m,
                        number_of_points=3,
                    ),
                    PointArray(
                        name="Line 2",
                        start=[0.1, 0.2, 0.3] * u.m,
                        end=[1.3, 1.5, 1.7] * u.m,
                        number_of_points=2,
                    ),
                ],
                output_fields=["primitiveVars", "Cp", vel_in_km_per_hr],
            ),
            ProbeOutput(
                name="prb point",
                entities=[
                    Point(
                        name="124",
                        location=[1, 1.02, 0.03] * u.cm,
                    ),
                    Point(
                        name="asdfg",
                        location=[0.0001, 0.02, 0.03] * u.m,
                    ),
                ],
                output_fields=["primitiveVars", "Cp", vel_in_km_per_hr],
            ),
            ProbeOutput(
                name="prb mix",
                entities=[
                    Point(
                        name="124",
                        location=[1, 1.02, 0.03] * u.cm,
                    ),
                    PointArray(
                        name="Line 1",
                        start=[0.1, 0.2, 0.3] * u.m,
                        end=[1.1, 1.2, 1.3] * u.m,
                        number_of_points=2,
                    ),
                ],
                output_fields=["primitiveVars", "Cp", vel_in_km_per_hr],
            ),
        ],
        {
            "outputFields": [],
            "monitors": {
                "prb line": {
                    "monitorLocations": {
                        "Line 1_0": [0.1, 0.2, 0.3],
                        "Line 1_1": [0.6, 0.7, 0.8],
                        "Line 1_2": [1.1, 1.2, 1.3],
                        "Line 2_0": [0.1, 0.2, 0.3],
                        "Line 2_1": [1.3, 1.5, 1.7],
                    },
                    "type": "probe",
                    "outputFields": ["Cp", "primitiveVars", "velocity_in_km_per_hr"],
                    "computeTimeAverages": False,
                    "animationFrequency": 1,
                    "animationFrequencyOffset": 0,
                },
                "prb point": {
                    "monitorLocations": {
                        "124": [0.01, 0.0102, 0.0003],
                        "asdfg": [0.0001, 0.02, 0.03],
                    },
                    "type": "probe",
                    "outputFields": ["Cp", "primitiveVars", "velocity_in_km_per_hr"],
                    "computeTimeAverages": False,
                    "animationFrequency": 1,
                    "animationFrequencyOffset": 0,
                },
                "prb mix": {
                    "monitorLocations": {
                        "124": [0.01, 0.0102, 0.0003],
                        "Line 1_0": [0.1, 0.2, 0.3],
                        "Line 1_1": [1.1, 1.2, 1.3],
                    },
                    "type": "probe",
                    "outputFields": ["Cp", "primitiveVars", "velocity_in_km_per_hr"],
                    "computeTimeAverages": False,
                    "animationFrequency": 1,
                    "animationFrequencyOffset": 0,
                },
            },
        },
    )


@pytest.fixture()
def surface_integral_output_config(vel_in_km_per_hr):
    return (
        [
            SurfaceIntegralOutput(  # Local
                name="prb 110",
                entities=[
                    Surface(name="surface1", private_attribute_full_name="zoneName/surface1"),
                    Surface(name="surface2"),
                ],
                output_fields=["My_field_1", vel_in_km_per_hr],
            ),
            SurfaceIntegralOutput(
                name="prb 122",
                entities=[
                    Surface(name="surface21"),
                    Surface(name="surface22"),
                ],
                output_fields=["My_field_2", vel_in_km_per_hr],
            ),  # Local
        ],
        {
            "monitors": {
                "prb 110": {
                    "animationFrequency": 1,
                    "animationFrequencyOffset": 0,
                    "computeTimeAverages": False,
                    "outputFields": ["My_field_1", "velocity_in_km_per_hr_integral"],
                    "surfaces": ["zoneName/surface1", "surface2"],
                    "type": "surfaceIntegral",
                },
                "prb 122": {
                    "animationFrequency": 1,
                    "animationFrequencyOffset": 0,
                    "computeTimeAverages": False,
                    "outputFields": ["My_field_2", "velocity_in_km_per_hr_integral"],
                    "surfaces": ["surface21", "surface22"],
                    "type": "surfaceIntegral",
                },
            },
            "outputFields": [],
        },
    )


def test_surface_probe_output(vel_in_km_per_hr):
    param_with_ref = (
        [
            SurfaceProbeOutput(
                name="SP-1",
                entities=[
                    Point(name="P1", location=[1, 1.02, 0.03] * u.cm),
                    Point(name="P2", location=[2, 1.01, 0.03] * u.m),
                ],
                target_surfaces=[
                    Surface(name="surface1", private_attribute_full_name="zoneA/surface1"),
                    Surface(name="surface2", private_attribute_full_name="zoneA/surface2"),
                ],
                output_fields=["Cp", "Cf", vel_in_km_per_hr],
            ),
            TimeAverageSurfaceProbeOutput(
                name="SP-2",
                entities=[
                    Point(name="P1", location=[1, 1.02, 0.03] * u.cm),
                    Point(name="P2", location=[2, 1.01, 0.03] * u.m),
                    Point(name="P3", location=[3, 1.02, 0.03] * u.m),
                ],
                target_surfaces=[
                    Surface(name="surface1", private_attribute_full_name="zoneB/surface1"),
                    Surface(name="surface2", private_attribute_full_name="zoneB/surface2"),
                ],
                output_fields=["Mach", "primitiveVars", "yPlus", vel_in_km_per_hr],
            ),
            SurfaceProbeOutput(
                name="SP-3",
                entities=[
                    PointArray(
                        name="PA1",
                        start=[0.1, 0.2, 0.3] * u.m,
                        end=[1.1, 1.2, 1.3] * u.m,
                        number_of_points=2,
                    ),
                    PointArray(
                        name="PA2",
                        start=[0.1, 0.2, 0.3] * u.m,
                        end=[1.3, 1.5, 1.7] * u.m,
                        number_of_points=2,
                    ),
                ],
                target_surfaces=[
                    Surface(name="surface1", private_attribute_full_name="zoneC/surface1"),
                    Surface(name="surface2", private_attribute_full_name="zoneC/surface2"),
                ],
                output_fields=["Mach", "primitiveVars", "yPlus", "my_own_field", vel_in_km_per_hr],
            ),
        ],
        {
            "outputFields": [],
            "monitors": {
                "SP-1": {
                    "monitorLocations": {"P1": [0.01, 0.0102, 0.0003], "P2": [2.0, 1.01, 0.03]},
                    "type": "surfaceProbe",
                    "outputFields": ["Cf", "Cp", "velocity_in_km_per_hr"],
                    "computeTimeAverages": False,
                    "animationFrequency": 1,
                    "animationFrequencyOffset": 0,
                    "surfacePatches": ["zoneA/surface1", "zoneA/surface2"],
                },
                "SP-3": {
                    "monitorLocations": {
                        "PA1_0": [0.1, 0.2, 0.3],
                        "PA1_1": [1.1, 1.2, 1.3],
                        "PA2_0": [0.1, 0.2, 0.3],
                        "PA2_1": [1.3, 1.5, 1.7],
                    },
                    "type": "surfaceProbe",
                    "outputFields": [
                        "Mach",
                        "my_own_field",
                        "primitiveVars",
                        "velocity_in_km_per_hr",
                        "yPlus",
                    ],
                    "computeTimeAverages": False,
                    "animationFrequency": 1,
                    "animationFrequencyOffset": 0,
                    "surfacePatches": ["zoneC/surface1", "zoneC/surface2"],
                },
                "SP-2": {
                    "monitorLocations": {
                        "P1": [0.01, 0.0102, 0.0003],
                        "P2": [2.0, 1.01, 0.03],
                        "P3": [3.0, 1.02, 0.03],
                    },
                    "type": "surfaceProbe",
                    "outputFields": ["Mach", "primitiveVars", "velocity_in_km_per_hr", "yPlus"],
                    "computeTimeAverages": True,
                    "animationFrequency": 1,
                    "animationFrequencyOffset": 0,
                    "animationFrequencyTimeAverage": 1,
                    "animationFrequencyTimeAverageOffset": 0,
                    "startAverageIntegrationStep": -1,
                    "surfacePatches": ["zoneB/surface1", "zoneB/surface2"],
                },
            },
        },
    )

    with SI_unit_system:
        param = SimulationParams(
            operating_condition=AerospaceCondition(),
            time_stepping=Unsteady(step_size=0.1 * u.s, steps=10),
            outputs=param_with_ref[0],
            user_defined_fields=[UserDefinedField(name="my_own_field", expression="1+1")],
        )
    param = param._preprocess(mesh_unit=1.0 * u.m, exclude=["models"])

    translated = {"boundaries": {}}
    translated = translate_output(param, translated)
    assert compare_values(param_with_ref[1], translated["surfaceMonitorOutput"])


def test_probe_output(
    probe_output_config,
    probe_output_with_point_array,
    surface_integral_output_config,
):
    ##:: monitorOutput
    with SI_unit_system:
        param = SimulationParams(
            operating_condition=AerospaceCondition(),
            time_stepping=Unsteady(step_size=0.1 * u.s, steps=10),
            outputs=probe_output_config[0],
        )
    param = param._preprocess(mesh_unit=1.0 * u.m, exclude=["models"])

    translated = {"boundaries": {}}
    translated = translate_output(param, translated)
    assert compare_values(probe_output_config[1], translated["monitorOutput"])

    ##:: monitorOutput with line probes
    with SI_unit_system:
        param = SimulationParams(outputs=probe_output_with_point_array[0])
    param = param._preprocess(mesh_unit=1.0 * u.m, exclude=["models"])

    translated = {"boundaries": {}}
    translated = translate_output(param, translated)
    assert compare_values(probe_output_with_point_array[1], translated["monitorOutput"])

    ##:: surfaceIntegral
    with SI_unit_system:
        param = SimulationParams(
            outputs=surface_integral_output_config[0],
            user_defined_fields=[
                UserDefinedField(name="My_field_1", expression="1+1"),
                UserDefinedField(name="My_field_2", expression="1+12"),
            ],
        )
    param = param._preprocess(mesh_unit=1 * u.m, exclude=["models"])

    translated = {"boundaries": {}}
    translated = translate_output(param, translated)
    assert compare_values(surface_integral_output_config[1], translated["monitorOutput"])

    ##:: surfaceIntegral and probeMonitor with global probe settings
    with SI_unit_system:
        param = SimulationParams(
            operating_condition=AerospaceCondition(),
            time_stepping=Unsteady(step_size=0.1 * u.s, steps=10),
            outputs=surface_integral_output_config[0] + probe_output_config[0],
            user_defined_fields=[
                UserDefinedField(name="My_field_1", expression="1+1"),
                UserDefinedField(name="My_field_2", expression="1+12"),
            ],
        )
    param = param._preprocess(mesh_unit=1 * u.m, exclude=["models"])

    translated = {"boundaries": {}}
    translated = translate_output(param, translated)

    ref = {
        "outputFields": [],
        "monitors": {
            "prb 10": {
                "monitorLocations": {"124": [0.01, 0.0102, 0.0003], "asdfg": [0.0001, 0.02, 0.03]},
                "type": "probe",
                "outputFields": ["Cp", "primitiveVars", "velocity_in_km_per_hr"],
                "computeTimeAverages": False,
                "animationFrequency": 1,
                "animationFrequencyOffset": 0,
            },
            "prb 12": {
                "monitorLocations": {"asnbgoujba": [0.1, 0.1002, 0.1003]},
                "type": "probe",
                "outputFields": ["Cp", "primitiveVars", "velocity_in_km_per_hr"],
                "computeTimeAverages": False,
                "animationFrequency": 1,
                "animationFrequencyOffset": 0,
            },
            "prb average": {
                "monitorLocations": {"a": [0.1, 0.1002, 0.1003]},
                "type": "probe",
                "outputFields": ["Cp", "T", "primitiveVars", "velocity_in_km_per_hr"],
                "computeTimeAverages": True,
                "animationFrequency": 1,
                "animationFrequencyOffset": 0,
                "animationFrequencyTimeAverage": 10,
                "animationFrequencyTimeAverageOffset": 0,
                "startAverageIntegrationStep": -1,
            },
            "prb 110": {
                "surfaces": ["zoneName/surface1", "surface2"],
                "type": "surfaceIntegral",
                "outputFields": ["My_field_1", "velocity_in_km_per_hr_integral"],
                "computeTimeAverages": False,
                "animationFrequency": 1,
                "animationFrequencyOffset": 0,
            },
            "prb 122": {
                "surfaces": ["surface21", "surface22"],
                "type": "surfaceIntegral",
                "outputFields": ["My_field_2", "velocity_in_km_per_hr_integral"],
                "computeTimeAverages": False,
                "animationFrequency": 1,
                "animationFrequencyOffset": 0,
            },
        },
    }
    assert compare_values(ref, translated["monitorOutput"])


@pytest.fixture()
def aeroacoustic_output_config():
    return (
        [
            AeroAcousticOutput(
                observers=[
                    Observer(position=[0.2, 0.02, 0.03] * u.m, group_name="0"),
                    Observer(position=[0.0001, 0.02, 0.03] * u.m, group_name="0"),
                ],
                write_per_surface_output=True,
                observer_time_step_size=0.1 * u.s,
            ),
        ],
        {
            "observers": [[0.2, 0.02, 0.03], [0.0001, 0.02, 0.03]],
            "writePerSurfaceOutput": True,
            "patchType": "solid",
            "observerTimeStepSize": 34.02940058082128,
            "startTime": 0.0,
            "newRun": False,
        },
    )


@pytest.fixture()
def aeroacoustic_output_permeable_config():
    return (
        [
            AeroAcousticOutput(
                observers=[
                    Observer(position=[1.2, 0.02, 0.03] * u.cm, group_name="0"),
                    Observer(position=[1, 0.02, 0.03] * u.cm, group_name="0"),
                ],
                patch_type="permeable",
                permeable_surfaces=[
                    Surface(
                        name="interface-A-B", private_attribute_full_name="zoneA/interface-A-B"
                    ),
                    Surface(
                        name="interface-A-C", private_attribute_full_name="zoneA/interface-A-C"
                    ),
                ],
            ),
        ],
        {
            "observers": [[0.012, 0.0002, 0.0003], [0.01, 0.0002, 0.0003]],
            "patchType": "permeable",
            "permeableSurfaces": ["zoneA/interface-A-B", "zoneA/interface-A-C"],
            "writePerSurfaceOutput": False,
            "startTime": 0.0,
            "newRun": False,
        },
    )


def test_acoustic_output(aeroacoustic_output_config, aeroacoustic_output_permeable_config):
    with SI_unit_system:
        param = SimulationParams(
            operating_condition=AerospaceCondition(),
            outputs=aeroacoustic_output_config[0],
            time_stepping=Unsteady(steps=1, step_size=0.1),
        )
    translated = {"boundaries": {}}
    param = param._preprocess(mesh_unit=1 * u.m, exclude=["models"])
    translated = translate_output(param, translated)

    assert compare_values(aeroacoustic_output_config[1], translated["aeroacousticOutput"])

    with SI_unit_system:
        param = SimulationParams(
            operating_condition=AerospaceCondition(),
            outputs=aeroacoustic_output_permeable_config[0],
            time_stepping=Unsteady(steps=1, step_size=0.1),
        )
    translated = {"boundaries": {}}
    param = param._preprocess(mesh_unit=1 * u.m, exclude=["models"])
    translated = translate_output(param, translated)

    assert compare_values(aeroacoustic_output_permeable_config[1], translated["aeroacousticOutput"])


def test_force_distribution_output():
    param_with_ref = (
        [
            ForceDistributionOutput(
                name="test_name",
                distribution_direction=[0.1, 0.9, 0.0],
            ),
        ],
        {
            "test_name": {
                "direction": [0.11043152607484655, 0.9938837346736189, 0.0],
                "type": "incremental",
            },
        },
    )

    with SI_unit_system:
        param = SimulationParams(outputs=param_with_ref[0])
    param = param._preprocess(mesh_unit=1.0 * u.m, exclude=["models"])

    translated = {}
    translated = translate_output(param, translated)
    assert compare_values(param_with_ref[1], translated["forceDistributionOutput"])


def test_time_averaged_force_distribution_output():
    param_with_ref = (
        [
            TimeAverageForceDistributionOutput(
                name="test_name",
                distribution_direction=[0.1, 0.9, 0.0],
            ),
            TimeAverageForceDistributionOutput(
                name="test_name2",
                distribution_direction=[1.0, 0.0, 0.0],
                distribution_type="cumulative",
                start_step=5,
            ),
        ],
        {
            "test_name": {
                "direction": [0.11043152607484655, 0.9938837346736189, 0.0],
                "type": "incremental",
                "startAverageIntegrationStep": -1,
            },
            "test_name2": {
                "direction": [1.0, 0.0, 0.0],
                "type": "cumulative",
                "startAverageIntegrationStep": 5,
            },
        },
    )

    with SI_unit_system:
        param = SimulationParams(
            outputs=param_with_ref[0], time_stepping=Unsteady(steps=1, step_size=0.1)
        )
    param = param._preprocess(mesh_unit=1.0 * u.m, exclude=["models"])

    translated = {}
    translated = translate_output(param, translated)
    assert compare_values(param_with_ref[1], translated["timeAveragedForceDistributionOutput"])


def test_surface_slice_output(vel_in_km_per_hr):
    param_with_ref = (
        [
            SurfaceSliceOutput(
                name="SS-1",
                entities=[
                    Slice(name="S1", origin=[1, 1.02, 0.03] * u.cm, normal=(0, 1, 0)),
                    Slice(name="S3", origin=[1, 1.01, 0.03] * u.cm, normal=(0, 1, 0)),
                ],
                target_surfaces=[
                    Surface(name="surface1", private_attribute_full_name="zoneA/surface1"),
                    Surface(name="surface2", private_attribute_full_name="zoneA/surface2"),
                ],
                output_fields=["Cp", "Cf", "primitiveVars", vel_in_km_per_hr],
                frequency=2,
            ),
            SurfaceSliceOutput(
                name="SS-2",
                entities=[
                    Slice(name="P1", origin=[1, 1.02, 0.03] * u.cm, normal=(0, 0, 1)),
                    Slice(name="P2", origin=[2, 1.01, 0.03] * u.m, normal=(0, 0, -1)),
                    Slice(name="P3", origin=[3, 1.02, 0.03] * u.m, normal=(0, 0, 1)),
                ],
                target_surfaces=[
                    Surface(name="surface1", private_attribute_full_name="zoneB/surface1"),
                    Surface(name="surface2", private_attribute_full_name="zoneB/surface2"),
                ],
                output_fields=["Mach", "primitiveVars", "yPlus", vel_in_km_per_hr],
            ),
        ],
        {
            "outputFields": [],
            "outputFormat": "paraview",
            "animationFrequency": 2,
            "animationFrequencyOffset": 0,
            "slices": [
                {
                    "name": "S1",
                    "sliceOrigin": [0.01, 0.0102, 0.0003],
                    "sliceNormal": [0.0, 1.0, 0.0],
                    "outputFields": ["Cf", "Cp", "primitiveVars", "velocity_in_km_per_hr"],
                    "surfacePatches": ["zoneA/surface1", "zoneA/surface2"],
                    "animationFrequency": 1,
                    "animationFrequencyOffset": 0,
                },
                {
                    "name": "S3",
                    "sliceOrigin": [0.01, 0.0101, 0.0003],
                    "sliceNormal": [0.0, 1.0, 0.0],
                    "outputFields": ["Cf", "Cp", "primitiveVars", "velocity_in_km_per_hr"],
                    "surfacePatches": ["zoneA/surface1", "zoneA/surface2"],
                    "animationFrequency": 1,
                    "animationFrequencyOffset": 0,
                },
                {
                    "name": "P1",
                    "sliceOrigin": [0.01, 0.0102, 0.0003],
                    "sliceNormal": [0.0, 0.0, 1.0],
                    "outputFields": ["Mach", "primitiveVars", "velocity_in_km_per_hr", "yPlus"],
                    "surfacePatches": ["zoneB/surface1", "zoneB/surface2"],
                    "animationFrequency": 1,
                    "animationFrequencyOffset": 0,
                },
                {
                    "name": "P2",
                    "sliceOrigin": [2.0, 1.01, 0.03],
                    "sliceNormal": [0.0, 0.0, -1.0],
                    "outputFields": ["Mach", "primitiveVars", "velocity_in_km_per_hr", "yPlus"],
                    "surfacePatches": ["zoneB/surface1", "zoneB/surface2"],
                    "animationFrequency": 1,
                    "animationFrequencyOffset": 0,
                },
                {
                    "name": "P3",
                    "sliceOrigin": [3.0, 1.02, 0.03],
                    "sliceNormal": [0.0, 0.0, 1.0],
                    "outputFields": ["Mach", "primitiveVars", "velocity_in_km_per_hr", "yPlus"],
                    "surfacePatches": ["zoneB/surface1", "zoneB/surface2"],
                    "animationFrequency": 1,
                    "animationFrequencyOffset": 0,
                },
            ],
        },
    )

    with SI_unit_system:
        param = SimulationParams(outputs=param_with_ref[0])
    param = param._preprocess(mesh_unit=1.0 * u.m, exclude=["models"])

    translated = {"boundaries": {}}
    translated = translate_output(param, translated)
    assert compare_values(param_with_ref[1], translated["surfaceSliceOutput"])


def test_dimensioned_output_fields_translation(vel_in_km_per_hr):
    """Test the translation of output fields from user-facing fields to solver fields."""

    with SI_unit_system:
        water = Water(
            name="h2o", density=1000 * u.kg / u.m**3, dynamic_viscosity=0.001 * u.kg / u.m / u.s
        )
        param = SimulationParams(
            operating_condition=LiquidOperatingCondition(
                velocity_magnitude=50 * u.m / u.s,
                reference_velocity_magnitude=100 * u.m / u.s,
                material=water,
            ),
            outputs=[
                VolumeOutput(
                    frequency=1,
                    output_format="both",
                    output_fields=[
                        "velocity",
                        "velocity_m_per_s",
                        "velocity_magnitude",
                        "velocity_magnitude_m_per_s",
                        "velocity_x_m_per_s",
                        "velocity_y_m_per_s",
                        "velocity_z_m_per_s",
                        "pressure",
                        "pressure_pa",
                        vel_in_km_per_hr,
                    ],
                ),
                SurfaceOutput(
                    entities=[Surface(name="surface11")],
                    output_fields=[
                        "wall_shear_stress_magnitude",
                        "wall_shear_stress_magnitude_pa",
                    ],
                ),
                SliceOutput(
                    name="my_slice",
                    entities=[Slice(name="my_slice", origin=[0, 0, 0], normal=(0, 1, 0))],
                    output_fields=["my_field"],
                ),
                IsosurfaceOutput(
                    name="my_isosurface",
                    entities=Isosurface(name="my_isosurface", iso_value=0.5, field="vorticity_y"),
                    output_fields=["my_field"],
                ),
            ],
            user_defined_fields=[
                UserDefinedField(
                    name="my_field",
                    expression="1+1",
                ),
            ],
        )

    solver_json = get_solver_json(param, mesh_unit=1.0 * u.m)
    expected_fields_v = [
        "velocity",
        "velocity_m_per_s",
        "velocity_magnitude",
        "velocity_magnitude_m_per_s",
        "velocity_x_m_per_s",
        "velocity_y_m_per_s",
        "velocity_z_m_per_s",
        "pressure",
        "pressure_pa",
        "velocity_in_km_per_hr",
    ]

    expected_fields_s = [
        "wall_shear_stress_magnitude",
        "wall_shear_stress_magnitude_pa",
    ]

    assert set(solver_json["volumeOutput"]["outputFields"]) == set(expected_fields_v)
    assert set(solver_json["surfaceOutput"]["surfaces"]["surface11"]["outputFields"]) == set(
        expected_fields_s
    )

    ref = {
        "userDefinedFields": [
            {"name": "my_field", "expression": "1+1", "from_user_variables": False},
            {
                "name": "pressure_pa",
                "expression": "double pressure_;double gamma = 1.4;pressure_ = (usingLiquidAsMaterial) ? (primitiveVars[4] - 1.0 / gamma) * (velocityScale * velocityScale) : primitiveVars[4];pressure_pa = pressure_ * 10000000.0;",
                "from_user_variables": False,
            },
            {
                "name": "velocity_in_km_per_hr",
                "expression": "double ___velocity[3];___velocity[0] = primitiveVars[1] * velocityScale;___velocity[1] = primitiveVars[2] * velocityScale;___velocity[2] = primitiveVars[3] * velocityScale;velocity_in_km_per_hr[0] = (___velocity[0] * 360.0); velocity_in_km_per_hr[1] = (___velocity[1] * 360.0); velocity_in_km_per_hr[2] = (___velocity[2] * 360.0);",
            },
            {
                "name": "velocity_m_per_s",
                "expression": "double velocity_[3];velocity_[0] = primitiveVars[1] * velocityScale;velocity_[1] = primitiveVars[2] * velocityScale;velocity_[2] = primitiveVars[3] * velocityScale;velocity_m_per_s[0] = velocity_[0] * 100.0;velocity_m_per_s[1] = velocity_[1] * 100.0;velocity_m_per_s[2] = velocity_[2] * 100.0;",
                "from_user_variables": False,
            },
            {
                "name": "velocity_magnitude",
                "expression": "double velocity[3];velocity[0] = primitiveVars[1];velocity[1] = primitiveVars[2];velocity[2] = primitiveVars[3];velocity_magnitude = magnitude(velocity) * velocityScale;",
                "from_user_variables": False,
            },
            {
                "name": "velocity_magnitude_m_per_s",
                "expression": "double velocity_magnitude;double velocity[3];velocity[0] = primitiveVars[1];velocity[1] = primitiveVars[2];velocity[2] = primitiveVars[3];velocity_magnitude = magnitude(velocity) * velocityScale;velocity_magnitude_m_per_s = velocity_magnitude * 100.0;",
                "from_user_variables": False,
            },
            {
                "name": "velocity_x_m_per_s",
                "expression": "double velocity_x;velocity_x = primitiveVars[1] * velocityScale;velocity_x_m_per_s = velocity_x * 100.0;",
                "from_user_variables": False,
            },
            {
                "name": "velocity_y_m_per_s",
                "expression": "double velocity_y;velocity_y = primitiveVars[2] * velocityScale;velocity_y_m_per_s = velocity_y * 100.0;",
                "from_user_variables": False,
            },
            {
                "name": "velocity_z_m_per_s",
                "expression": "double velocity_z;velocity_z = primitiveVars[3] * velocityScale;velocity_z_m_per_s = velocity_z * 100.0;",
                "from_user_variables": False,
            },
            {
                "name": "vorticity_y",
                "expression": "vorticity_y = (gradPrimitive[1][2] - gradPrimitive[3][0]) * velocityScale;",
                "from_user_variables": False,
            },
            {
                "name": "wall_shear_stress_magnitude",
                "expression": "wall_shear_stress_magnitude = magnitude(wallShearStress) * (velocityScale * velocityScale);",
                "from_user_variables": False,
            },
            {
                "name": "wall_shear_stress_magnitude_pa",
                "expression": "double wall_shear_stress_magnitude;wall_shear_stress_magnitude = magnitude(wallShearStress) * (velocityScale * velocityScale);wall_shear_stress_magnitude_pa = wall_shear_stress_magnitude * 10000000.0;",
                "from_user_variables": False,
            },
        ]
    }
    print(json.dumps(solver_json["userDefinedFields"], indent=2))
    translated_udfs = sorted(solver_json["userDefinedFields"], key=lambda x: x["name"])
    ref_udfs = sorted(ref["userDefinedFields"], key=lambda x: x["name"])
    print(">>>", translated_udfs)
    assert compare_values(translated_udfs, ref_udfs)


@pytest.fixture()
def streamline_output_config():
    return (
        [
            StreamlineOutput(
                entities=[
                    Point(name="point_streamline", location=(0.0, 1.0, 0.04) * u.m),
                    PointArray(
                        name="pointarray_streamline",
                        start=(0.0, 0.0, 0.2) * u.m,
                        end=(0.0, 1.0, 0.2) * u.m,
                        number_of_points=20,
                    ),
                    PointArray2D(
                        name="pointarray2d_streamline",
                        origin=(0.0, 0.0, -0.2) * u.m,
                        u_axis_vector=(0.0, 1.4, 0.0) * u.m,
                        v_axis_vector=(0.0, 0.0, 0.4) * u.m,
                        u_number_of_points=10,
                        v_number_of_points=10,
                    ),
                ]
            )
        ],
        {
            "PointArrays": [
                {
                    "end": [0.0, 1.0, 0.2],
                    "name": "pointarray_streamline",
                    "numberOfPoints": 20,
                    "start": [0.0, 0.0, 0.2],
                }
            ],
            "PointArrays2D": [
                {
                    "name": "pointarray2d_streamline",
                    "origin": [0.0, 0.0, -0.2],
                    "uAxisVector": [0.0, 1.4, 0.0],
                    "uNumberOfPoints": 10,
                    "vAxisVector": [0.0, 0.0, 0.4],
                    "vNumberOfPoints": 10,
                }
            ],
            "Points": [{"location": [0.0, 1.0, 0.04], "name": "point_streamline"}],
        },
    )


def test_streamline_output(streamline_output_config):
    with SI_unit_system:
        param = SimulationParams(
            operating_condition=AerospaceCondition(),
            outputs=streamline_output_config[0],
            time_stepping=Unsteady(step_size=0.1 * u.s, steps=100),
        )
    translated = {"boundaries": {}}
    param = param._preprocess(mesh_unit=1 * u.m, exclude=["models"])
    translated = translate_output(param, translated)


@pytest.fixture()
def imported_surface_output_config(vel_in_km_per_hr):
    return (
        [
            SurfaceOutput(
                output_fields=[
                    vel_in_km_per_hr,
                ],
                surfaces=[
                    ImportedSurface(name="normal", file_name="rectangle_normal.cgns"),
                    ImportedSurface(name="oblique", file_name="rectangle_oblique.cgns"),
                ],
            ),
        ],
        {
            "animationFrequency": -1,
            "animationFrequencyOffset": 0,
            "outputFields": [],
            "outputFormat": "paraview",
            "surfaces": {
                "normal": {
                    "meshFile": "rectangle_normal.cgns",
                    "outputFields": ["velocity_in_km_per_hr"],
                },
                "oblique": {
                    "meshFile": "rectangle_oblique.cgns",
                    "outputFields": ["velocity_in_km_per_hr"],
                },
            },
        },
    )


@pytest.fixture()
def time_average_imported_surface_output_config(vel_in_km_per_hr):
    return (
        [
            TimeAverageSurfaceOutput(
                output_fields=[
                    vel_in_km_per_hr,
                ],
                surfaces=[
                    ImportedSurface(name="normal", file_name="rectangle_normal.cgns"),
                    ImportedSurface(name="oblique", file_name="rectangle_oblique.cgns"),
                ],
                frequency=4,
                start_step=10,
            ),
        ],
        {
            "animationFrequencyTimeAverage": 4,
            "animationFrequencyTimeAverageOffset": 0,
            "outputFields": [],
            "outputFormat": "paraview",
            "startAverageIntegrationStep": 10,
            "surfaces": {
                "normal": {
                    "meshFile": "rectangle_normal.cgns",
                    "outputFields": ["velocity_in_km_per_hr"],
                },
                "oblique": {
                    "meshFile": "rectangle_oblique.cgns",
                    "outputFields": ["velocity_in_km_per_hr"],
                },
            },
        },
    )


@pytest.fixture()
def imported_surface_integral_output_config(vel_in_km_per_hr):
    return (
        [
            SurfaceIntegralOutput(
                name="MassFlowRateImportedSurface",
                output_fields=[UserVariable(name="MassFluxProjected", value=vel_in_km_per_hr)],
                surfaces=[
                    ImportedSurface(name="normal", file_name="rectangle_normal.cgns"),
                    ImportedSurface(name="oblique", file_name="rectangle_oblique.cgns"),
                ],
            ),
        ],
        {
            "animationFrequency": -1,
            "animationFrequencyOffset": 0,
            "surfaces": {
                "normal": {
                    "meshFile": "rectangle_normal.cgns",
                    "outputFields": ["MassFluxProjected_integral"],
                },
                "oblique": {
                    "meshFile": "rectangle_oblique.cgns",
                    "outputFields": ["MassFluxProjected_integral"],
                },
            },
        },
    )


def test_imported_surface_output(
    imported_surface_output_config,
    imported_surface_integral_output_config,
    time_average_imported_surface_output_config,
):
    with SI_unit_system:
        param = SimulationParams(
            operating_condition=AerospaceCondition(),
            outputs=imported_surface_integral_output_config[0]
            + imported_surface_output_config[0]
            + time_average_imported_surface_output_config[0],
            time_stepping=Unsteady(step_size=0.1 * u.s, steps=100),
        )
    translated = {"boundaries": {}}
    param = param._preprocess(mesh_unit=1 * u.m, exclude=["models"])
    translated = translate_output(param, translated)
    assert compare_values(imported_surface_output_config[1], translated["importedSurfaceOutput"])
    assert compare_values(
        imported_surface_integral_output_config[1], translated["importedSurfaceIntegralOutput"]
    )
    assert compare_values(
        time_average_imported_surface_output_config[1],
        translated["timeAverageImportedSurfaceOutput"],
    )


def test_imported_surface_with_coordinate_system_transformation():
    """Test that ImportedSurface with coordinate system assignment includes transformation matrix in output JSON."""
    with SI_unit_system:
        # Create ImportedSurface entities with explicit IDs
        imported_surface_1 = ImportedSurface(
            name="surface1", file_name="surface1.cgns", private_attribute_id="surface1_id"
        )
        imported_surface_2 = ImportedSurface(
            name="surface2", file_name="surface2.cgns", private_attribute_id="surface2_id"
        )

        # Create coordinate system with simple translation
        cs = CoordinateSystem(
            name="test_frame",
            translation=(10, 20, 30) * u.m,
        )

        # Create entity registry and register entities
        entity_registry = EntityRegistry()
        entity_registry.register(imported_surface_1)
        entity_registry.register(imported_surface_2)

        # Create CoordinateSystemManager and assign coordinate system to first surface
        manager = CoordinateSystemManager()
        manager.add(coordinate_system=cs)
        manager.assign(entities=imported_surface_1, coordinate_system=cs)

        # Create coordinate system status
        cs_status = manager._to_status()

        # Create a user variable for SurfaceIntegralOutput
        mass_flux_var = UserVariable(name="TestSurfaceIntegral", value=solution.velocity)

        # Create SimulationParams with ImportedSurface outputs and coordinate system status
        param = SimulationParams(
            operating_condition=AerospaceCondition(),
            outputs=[
                SurfaceOutput(
                    output_fields=["velocity"],
                    surfaces=[imported_surface_1, imported_surface_2],
                ),
                SurfaceIntegralOutput(
                    name="ImportedSurfaceIntegral",
                    output_fields=[mass_flux_var],
                    surfaces=[imported_surface_1, imported_surface_2],
                ),
            ],
            time_stepping=Unsteady(step_size=0.1 * u.s, steps=100),
            private_attribute_asset_cache=AssetCache(coordinate_system_status=cs_status),
        )

    # Preprocess and translate
    translated = {"boundaries": {}}
    param = param._preprocess(mesh_unit=1 * u.m, exclude=["models"])
    translated = translate_output(param, translated)

    # Verify the output structure
    assert "importedSurfaceOutput" in translated
    assert "surfaces" in translated["importedSurfaceOutput"]

    # Check surface1 (with coordinate system) has transformation matrix
    surface1_output = translated["importedSurfaceOutput"]["surfaces"]["surface1"]
    assert "meshFile" in surface1_output
    assert surface1_output["meshFile"] == "surface1.cgns"
    assert "transformationMatrix" in surface1_output

    # Verify the transformation matrix is correct (identity rotation + translation [10, 20, 30])
    expected_matrix = np.array(
        [[1, 0, 0, 10], [0, 1, 0, 20], [0, 0, 1, 30]], dtype=np.float64
    ).tolist()
    np.testing.assert_allclose(
        surface1_output["transformationMatrix"],
        expected_matrix,
        atol=1e-10,
        err_msg="Transformation matrix does not match expected translation",
    )

    # Check surface2 (without coordinate system) has NO transformation matrix
    surface2_output = translated["importedSurfaceOutput"]["surfaces"]["surface2"]
    assert "meshFile" in surface2_output
    assert surface2_output["meshFile"] == "surface2.cgns"
    assert "transformationMatrix" not in surface2_output

    # Verify the same for importedSurfaceIntegralOutput
    assert "importedSurfaceIntegralOutput" in translated
    assert "surfaces" in translated["importedSurfaceIntegralOutput"]

    surface1_integral_output = translated["importedSurfaceIntegralOutput"]["surfaces"]["surface1"]
    assert "transformationMatrix" in surface1_integral_output
    np.testing.assert_allclose(
        surface1_integral_output["transformationMatrix"],
        expected_matrix,
        atol=1e-10,
    )

    surface2_integral_output = translated["importedSurfaceIntegralOutput"]["surfaces"]["surface2"]
    assert "transformationMatrix" not in surface2_integral_output


def test_imported_surface_with_rotation_and_translation():
    """Test that ImportedSurface with coordinate system including rotation outputs correct matrix."""
    with SI_unit_system:
        # Create ImportedSurface entity with explicit ID
        imported_surface = ImportedSurface(
            name="rotated_surface",
            file_name="rotated.cgns",
            private_attribute_id="rotated_surface_id",
        )

        # Create coordinate system with 90 degree rotation around Z and translation
        cs = CoordinateSystem(
            name="rotated_frame",
            axis_of_rotation=(0, 0, 1),
            angle_of_rotation=90 * u.deg,
            translation=(5, 10, 15) * u.m,
        )

        # Create entity registry and register entity
        entity_registry = EntityRegistry()
        entity_registry.register(imported_surface)

        # Create CoordinateSystemManager and assign
        manager = CoordinateSystemManager()
        manager.add(coordinate_system=cs)
        manager.assign(entities=imported_surface, coordinate_system=cs)

        cs_status = manager._to_status()

        # Create SimulationParams
        param = SimulationParams(
            operating_condition=AerospaceCondition(),
            outputs=[
                SurfaceOutput(
                    output_fields=["velocity"],
                    surfaces=[imported_surface],
                ),
            ],
            time_stepping=Unsteady(step_size=0.1 * u.s, steps=100),
            private_attribute_asset_cache=AssetCache(coordinate_system_status=cs_status),
        )

    # Translate
    translated = {"boundaries": {}}
    param = param._preprocess(mesh_unit=1 * u.m, exclude=["models"])
    translated = translate_output(param, translated)

    # Verify the transformation matrix
    surface_output = translated["importedSurfaceOutput"]["surfaces"]["rotated_surface"]
    assert "transformationMatrix" in surface_output

    # Expected: 90 degree rotation around Z + translation [5, 10, 15]
    # Rotation matrix for 90 deg around Z: [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    expected_matrix = np.array([[0, -1, 0, 5], [1, 0, 0, 10], [0, 0, 1, 15]], dtype=np.float64)

    np.testing.assert_allclose(
        surface_output["transformationMatrix"],
        expected_matrix.tolist(),
        atol=1e-10,
        err_msg="Transformation matrix with rotation does not match expected",
    )
