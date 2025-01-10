import json

import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.outputs.output_entities import Point, PointArray
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
    SurfaceProbeOutput,
    SurfaceSliceOutput,
    TimeAverageProbeOutput,
    TimeAverageSurfaceOutput,
    TimeAverageSurfaceProbeOutput,
    TimeAverageVolumeOutput,
    UserDefinedField,
    VolumeOutput,
)
from flow360.component.simulation.primitives import Surface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.translator.solver_translator import translate_output
from flow360.component.simulation.unit_system import SI_unit_system


@pytest.fixture()
def volume_output_config():
    return (
        VolumeOutput(
            frequency=1,
            frequency_offset=2,
            output_format="both",
            output_fields=["primitiveVars", "betMetrics", "qcriterion"],
        ),
        {
            "animationFrequency": 1,
            "animationFrequencyOffset": 2,
            "animationFrequencyTimeAverage": -1,
            "animationFrequencyTimeAverageOffset": 0,
            "computeTimeAverages": False,
            "outputFields": ["primitiveVars", "betMetrics", "qcriterion"],
            "outputFormat": "paraview,tecplot",
            "startAverageIntegrationStep": -1,
        },
    )


@pytest.fixture()
def avg_volume_output_config():
    return (
        TimeAverageVolumeOutput(
            frequency=11,
            frequency_offset=12,
            output_format="both",
            output_fields=["primitiveVars", "betMetrics", "qcriterion"],
            start_step=1,
        ),
        {
            "animationFrequency": -1,
            "animationFrequencyOffset": 0,
            "animationFrequencyTimeAverage": 11,
            "animationFrequencyTimeAverageOffset": 12,
            "computeTimeAverages": True,
            "outputFields": ["primitiveVars", "betMetrics", "qcriterion"],
            "outputFormat": "paraview,tecplot",
            "startAverageIntegrationStep": 1,
        },
    )


def test_volume_output(volume_output_config, avg_volume_output_config):
    import json

    ##:: volumeOutput only
    with SI_unit_system:
        param = SimulationParams(outputs=[volume_output_config[0]])
    translated = {"boundaries": {}}
    translated = translate_output(param, translated)
    assert sorted(volume_output_config[1].items()) == sorted(translated["volumeOutput"].items())

    ##:: timeAverageVolumeOutput only
    with SI_unit_system:
        param = SimulationParams(outputs=[avg_volume_output_config[0]])
    translated = {"boundaries": {}}
    translated = translate_output(param, translated)
    assert sorted(avg_volume_output_config[1].items()) == sorted(translated["volumeOutput"].items())

    ##:: timeAverageVolumeOutput and volumeOutput
    with SI_unit_system:
        param = SimulationParams(outputs=[volume_output_config[0], avg_volume_output_config[0]])
    translated = {"boundaries": {}}
    translated = translate_output(param, translated)
    ref = {
        "volumeOutput": {
            "animationFrequency": 1,
            "animationFrequencyOffset": 2,
            "animationFrequencyTimeAverage": 11,
            "animationFrequencyTimeAverageOffset": 12,
            "computeTimeAverages": True,
            "outputFields": ["primitiveVars", "betMetrics", "qcriterion"],
            "outputFormat": "paraview,tecplot",
            "startAverageIntegrationStep": 1,
        }
    }
    assert sorted(ref["volumeOutput"].items()) == sorted(translated["volumeOutput"].items())


@pytest.fixture()
def surface_output_config():
    return (
        [
            SurfaceOutput(  # Local
                entities=[Surface(name="surface1"), Surface(name="surface2")],
                output_fields=["Cp"],
                output_format="tecplot",
                frequency=123,
                frequency_offset=321,
            ),
            SurfaceOutput(  # Local
                entities=[Surface(name="surface11"), Surface(name="surface22")],
                frequency=123,
                frequency_offset=321,
                output_fields=["T"],
                output_format="tecplot",
            ),
        ],
        {
            "animationFrequency": 123,
            "animationFrequencyOffset": 321,
            "animationFrequencyTimeAverage": -1,
            "animationFrequencyTimeAverageOffset": 0,
            "computeTimeAverages": False,
            "outputFields": [],
            "outputFormat": "tecplot",
            "startAverageIntegrationStep": -1,
            "surfaces": {
                "surface1": {"outputFields": ["Cp"]},
                "surface11": {"outputFields": ["T"]},
                "surface2": {"outputFields": ["Cp"]},
                "surface22": {"outputFields": ["T"]},
            },
            "writeSingleFile": False,
        },
    )


@pytest.fixture()
def avg_surface_output_config():
    return [
        TimeAverageSurfaceOutput(  # Local
            entities=[Surface(name="surface1"), Surface(name="surface2")],
            output_fields=["Cp"],
        ),
        TimeAverageSurfaceOutput(  # Local
            entities=[Surface(name="surface3")],
            output_fields=["T"],
        ),
    ]


def test_surface_output(
    surface_output_config,
    avg_surface_output_config,
):
    ##:: surfaceOutput with No global settings
    with SI_unit_system:
        param = SimulationParams(outputs=surface_output_config[0])
    translated = {"boundaries": {}}
    translated = translate_output(param, translated)
    assert sorted(surface_output_config[1].items()) == sorted(translated["surfaceOutput"].items())

    ##:: timeAverageSurfaceOutput and surfaceOutput
    with SI_unit_system:
        param = SimulationParams(outputs=surface_output_config[0] + avg_surface_output_config)
    translated = {"boundaries": {}}
    translated = translate_output(param, translated)
    ref = {
        "animationFrequency": 123,
        "animationFrequencyOffset": 321,
        "animationFrequencyTimeAverage": -1,
        "animationFrequencyTimeAverageOffset": 0,
        "computeTimeAverages": True,
        "outputFields": [],
        "outputFormat": "paraview",
        "startAverageIntegrationStep": -1,
        "surfaces": {
            "surface1": {"outputFields": ["Cp"]},
            "surface11": {"outputFields": ["T"]},
            "surface2": {"outputFields": ["Cp"]},
            "surface22": {"outputFields": ["T"]},
            "surface3": {"outputFields": ["T"]},
        },
        "writeSingleFile": False,
    }
    assert sorted(ref.items()) == sorted(translated["surfaceOutput"].items())


@pytest.fixture()
def sliceoutput_config():
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
                output_fields=["Cp"],
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
                output_fields=["T", "primitiveVars"],
            ),
        ],
        {
            "animationFrequency": 33,
            "animationFrequencyOffset": 22,
            "animationFrequencyTimeAverage": -1,
            "animationFrequencyTimeAverageOffset": 0,
            "startAverageIntegrationStep": -1,
            "computeTimeAverages": False,
            "outputFields": [],
            "outputFormat": "tecplot",
            "slices": {
                "slice01": {
                    "outputFields": ["T", "primitiveVars"],
                    "sliceNormal": [1.0, 0.0, 0.0],
                    "sliceOrigin": [10.02, 10.03, 10.04],
                },
                "slice02": {
                    "outputFields": ["T", "primitiveVars"],
                    "sliceNormal": [0.6, 0.0, 0.8],
                    "sliceOrigin": [6.12, 6.13, 6.14],
                },
                "slice10": {
                    "outputFields": ["Cp"],
                    "sliceNormal": [0.0, 1.0, 0.0],
                    "sliceOrigin": [0.02, 0.03, 0.04],
                },
                "slice20": {
                    "outputFields": ["Cp"],
                    "sliceNormal": [0.6, 0.8, 0.0],
                    "sliceOrigin": [0.12, 0.13, 0.14],
                },
            },
        },
    )


def test_slice_output(
    sliceoutput_config,
):
    ##:: sliceOutput with NO global settings
    with SI_unit_system:
        param = SimulationParams(outputs=sliceoutput_config[0])
    param = param.preprocess(1.0 * u.m, exclude=["models"])
    translated = {"boundaries": {}}
    translated = translate_output(param, translated)

    assert sorted(sliceoutput_config[1].items()) == sorted(translated["sliceOutput"].items())


@pytest.fixture()
def isosurface_output_config():
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
                ],
                output_fields=["Cp"],
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
                output_fields=["T", "primitiveVars"],
            ),
        ],
        {
            "animationFrequency": 332,
            "animationFrequencyOffset": 222,
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

    assert sorted(isosurface_output_config[1].items()) == sorted(
        translated["isoSurfaceOutput"].items()
    )


@pytest.fixture()
def probe_output_config():
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
                output_fields=["primitiveVars", "Cp"],
            ),
            ProbeOutput(  # Local
                name="prb 12",
                entities=[
                    Point(
                        name="asnbgoujba",
                        location=[10, 10.02, 10.03] * u.cm,
                    ),
                ],
                output_fields=["primitiveVars", "Cp"],
            ),
            TimeAverageProbeOutput(  # Local
                name="prb average",
                entities=[
                    Point(
                        name="a",
                        location=[10, 10.02, 10.03] * u.cm,
                    ),
                ],
                output_fields=["primitiveVars", "Cp", "T"],
                frequency=10,
            ),
        ],
        {
            "monitors": {
                "prb 10": {
                    "animationFrequency": 1,
                    "animationFrequencyOffset": 0,
                    "computeTimeAverages": False,
                    "start": [[1e-2, 1.02e-2, 0.0003], [0.0001, 0.02, 0.03]],
                    "end": [[1e-2, 1.02e-2, 0.0003], [0.0001, 0.02, 0.03]],
                    "numberOfPoints": [1, 1],
                    "outputFields": ["primitiveVars", "Cp"],
                    "type": "lineProbe",
                },
                "prb 12": {
                    "animationFrequency": 1,
                    "animationFrequencyOffset": 0,
                    "computeTimeAverages": False,
                    "start": [[10e-2, 10.02e-2, 10.03e-2]],
                    "end": [[10e-2, 10.02e-2, 10.03e-2]],
                    "numberOfPoints": [1],
                    "outputFields": ["primitiveVars", "Cp"],
                    "type": "lineProbe",
                },
                "prb average": {
                    "animationFrequency": 1,
                    "animationFrequencyOffset": 0,
                    "animationFrequencyTimeAverage": 10,
                    "animationFrequencyTimeAverageOffset": 0,
                    "startAverageIntegrationStep": -1,
                    "computeTimeAverages": True,
                    "start": [[10e-2, 10.02e-2, 10.03e-2]],
                    "end": [[10e-2, 10.02e-2, 10.03e-2]],
                    "numberOfPoints": [1],
                    "outputFields": ["primitiveVars", "Cp", "T"],
                    "type": "lineProbe",
                },
            },
            "outputFields": [],
        },
    )


@pytest.fixture()
def probe_output_with_point_array():
    return (
        [
            ProbeOutput(
                name="prb line",
                entities=[
                    PointArray(
                        name="Line 1",
                        start=[0.1, 0.2, 0.3] * u.m,
                        end=[1.1, 1.2, 1.3] * u.m,
                        number_of_points=5,
                    ),
                    PointArray(
                        name="Line 2",
                        start=[0.1, 0.2, 0.3] * u.m,
                        end=[1.3, 1.5, 1.7] * u.m,
                        number_of_points=7,
                    ),
                ],
                output_fields=["primitiveVars", "Cp"],
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
                output_fields=["primitiveVars", "Cp"],
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
                        number_of_points=5,
                    ),
                ],
                output_fields=["primitiveVars", "Cp"],
            ),
        ],
        {
            "monitors": {
                "prb line": {
                    "start": [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]],
                    "end": [[1.1, 1.2, 1.3], [1.3, 1.5, 1.7]],
                    "numberOfPoints": [5, 7],
                    "outputFields": ["primitiveVars", "Cp"],
                    "animationFrequency": 1,
                    "animationFrequencyOffset": 0,
                    "computeTimeAverages": False,
                    "type": "lineProbe",
                },
                "prb point": {
                    "start": [[1e-2, 1.02e-2, 0.0003], [0.0001, 0.02, 0.03]],
                    "end": [[1e-2, 1.02e-2, 0.0003], [0.0001, 0.02, 0.03]],
                    "numberOfPoints": [1, 1],
                    "outputFields": ["primitiveVars", "Cp"],
                    "animationFrequency": 1,
                    "animationFrequencyOffset": 0,
                    "computeTimeAverages": False,
                    "type": "lineProbe",
                },
                "prb mix": {
                    "start": [[1e-2, 1.02e-2, 0.0003], [0.1, 0.2, 0.3]],
                    "end": [[1e-2, 1.02e-2, 0.0003], [1.1, 1.2, 1.3]],
                    "numberOfPoints": [1, 5],
                    "outputFields": ["primitiveVars", "Cp"],
                    "animationFrequency": 1,
                    "animationFrequencyOffset": 0,
                    "computeTimeAverages": False,
                    "type": "lineProbe",
                },
            },
            "outputFields": [],
        },
    )


@pytest.fixture()
def surface_integral_output_config():
    return (
        [
            SurfaceIntegralOutput(  # Local
                name="prb 110",
                entities=[
                    Surface(name="surface1", private_attribute_full_name="zoneName/surface1"),
                    Surface(name="surface2"),
                ],
                output_fields=["My_field_1"],
            ),
            SurfaceIntegralOutput(
                name="prb 122",
                entities=[
                    Surface(name="surface21"),
                    Surface(name="surface22"),
                ],
                output_fields=["My_field_2"],
            ),  # Local
        ],
        {
            "monitors": {
                "prb 110": {
                    "animationFrequency": 1,
                    "animationFrequencyOffset": 0,
                    "computeTimeAverages": False,
                    "outputFields": ["My_field_1"],
                    "surfaces": ["zoneName/surface1", "surface2"],
                    "type": "surfaceIntegral",
                },
                "prb 122": {
                    "animationFrequency": 1,
                    "animationFrequencyOffset": 0,
                    "computeTimeAverages": False,
                    "outputFields": ["My_field_2"],
                    "surfaces": ["surface21", "surface22"],
                    "type": "surfaceIntegral",
                },
            },
            "outputFields": [],
        },
    )


def test_surface_probe_output():
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
                output_fields=["Cp", "Cf"],
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
                output_fields=["Mach", "primitiveVars", "yPlus"],
            ),
            SurfaceProbeOutput(
                name="SP-3",
                entities=[
                    PointArray(
                        name="PA1",
                        start=[0.1, 0.2, 0.3] * u.m,
                        end=[1.1, 1.2, 1.3] * u.m,
                        number_of_points=5,
                    ),
                    PointArray(
                        name="PA2",
                        start=[0.1, 0.2, 0.3] * u.m,
                        end=[1.3, 1.5, 1.7] * u.m,
                        number_of_points=7,
                    ),
                ],
                target_surfaces=[
                    Surface(name="surface1", private_attribute_full_name="zoneC/surface1"),
                    Surface(name="surface2", private_attribute_full_name="zoneC/surface2"),
                ],
                output_fields=["Mach", "primitiveVars", "yPlus", "my_own_field"],
            ),
        ],
        {
            "monitors": {
                "SP-1": {
                    "animationFrequency": 1,
                    "animationFrequencyOffset": 0,
                    "computeTimeAverages": False,
                    "outputFields": ["Cp", "Cf"],
                    "surfacePatches": ["zoneA/surface1", "zoneA/surface2"],
                    "start": [[1e-2, 1.02e-2, 0.0003], [2.0, 1.01, 0.03]],
                    "end": [[1e-2, 1.02e-2, 0.0003], [2.0, 1.01, 0.03]],
                    "numberOfPoints": [1, 1],
                    "type": "lineProbe",
                },
                "SP-2": {
                    "animationFrequency": 1,
                    "animationFrequencyOffset": 0,
                    "animationFrequencyTimeAverage": 1,
                    "animationFrequencyTimeAverageOffset": 0,
                    "startAverageIntegrationStep": -1,
                    "computeTimeAverages": True,
                    "outputFields": ["Mach", "primitiveVars", "yPlus"],
                    "surfacePatches": ["zoneB/surface1", "zoneB/surface2"],
                    "start": [
                        [1e-2, 1.02e-2, 0.0003],
                        [2.0, 1.01, 0.03],
                        [3.0, 1.02, 0.03],
                    ],
                    "end": [
                        [1e-2, 1.02e-2, 0.0003],
                        [2.0, 1.01, 0.03],
                        [3.0, 1.02, 0.03],
                    ],
                    "numberOfPoints": [1, 1, 1],
                    "type": "lineProbe",
                },
                "SP-3": {
                    "animationFrequency": 1,
                    "animationFrequencyOffset": 0,
                    "computeTimeAverages": False,
                    "outputFields": ["Mach", "primitiveVars", "yPlus", "my_own_field"],
                    "surfacePatches": ["zoneC/surface1", "zoneC/surface2"],
                    "start": [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]],
                    "end": [[1.1, 1.2, 1.3], [1.3, 1.5, 1.7]],
                    "numberOfPoints": [5, 7],
                    "type": "lineProbe",
                },
            },
            "outputFields": [],
        },
    )

    with SI_unit_system:
        param = SimulationParams(
            outputs=param_with_ref[0],
            user_defined_fields=[UserDefinedField(name="my_own_field", expression="1+1")],
        )
    param = param.preprocess(mesh_unit=1.0 * u.m, exclude=["models"])

    translated = {"boundaries": {}}
    translated = translate_output(param, translated)
    assert sorted(param_with_ref[1].items()) == sorted(translated["surfaceMonitorOutput"].items())


def test_monitor_output(
    probe_output_config,
    probe_output_with_point_array,
    surface_integral_output_config,
):
    ##:: monitorOutput
    with SI_unit_system:
        param = SimulationParams(outputs=probe_output_config[0])
    param = param.preprocess(mesh_unit=1.0 * u.m, exclude=["models"])

    translated = {"boundaries": {}}
    translated = translate_output(param, translated)
    assert sorted(probe_output_config[1].items()) == sorted(translated["monitorOutput"].items())

    ##:: monitorOutput with line probes
    with SI_unit_system:
        param = SimulationParams(outputs=probe_output_with_point_array[0])
    param = param.preprocess(mesh_unit=1.0 * u.m, exclude=["models"])

    translated = {"boundaries": {}}
    translated = translate_output(param, translated)
    assert sorted(probe_output_with_point_array[1].items()) == sorted(
        translated["monitorOutput"].items()
    )

    ##:: surfaceIntegral
    with SI_unit_system:
        param = SimulationParams(
            outputs=surface_integral_output_config[0],
            user_defined_fields=[
                UserDefinedField(name="My_field_1", expression="1+1"),
                UserDefinedField(name="My_field_2", expression="1+12"),
            ],
        )
    param = param.preprocess(mesh_unit=1 * u.m, exclude=["models"])

    translated = {"boundaries": {}}
    translated = translate_output(param, translated)
    assert sorted(surface_integral_output_config[1].items()) == sorted(
        translated["monitorOutput"].items()
    )

    ##:: surfaceIntegral and probeMonitor with global probe settings
    with SI_unit_system:
        param = SimulationParams(
            outputs=surface_integral_output_config[0] + probe_output_config[0],
            user_defined_fields=[
                UserDefinedField(name="My_field_1", expression="1+1"),
                UserDefinedField(name="My_field_2", expression="1+12"),
            ],
        )
    param = param.preprocess(mesh_unit=1 * u.m, exclude=["models"])

    translated = {"boundaries": {}}
    translated = translate_output(param, translated)
    ref = {
        "monitors": {
            "prb 10": {
                "animationFrequency": 1,
                "animationFrequencyOffset": 0,
                "computeTimeAverages": False,
                "start": [[1e-2, 1.02e-2, 0.0003], [0.0001, 0.02, 0.03]],
                "end": [[1e-2, 1.02e-2, 0.0003], [0.0001, 0.02, 0.03]],
                "numberOfPoints": [1, 1],
                "outputFields": ["primitiveVars", "Cp"],
                "type": "lineProbe",
            },
            "prb 110": {
                "animationFrequency": 1,
                "animationFrequencyOffset": 0,
                "computeTimeAverages": False,
                "outputFields": ["My_field_1"],
                "surfaces": ["zoneName/surface1", "surface2"],
                "type": "surfaceIntegral",
            },
            "prb 12": {
                "animationFrequency": 1,
                "animationFrequencyOffset": 0,
                "computeTimeAverages": False,
                "start": [[10e-2, 10.02e-2, 10.03e-2]],
                "end": [[10e-2, 10.02e-2, 10.03e-2]],
                "numberOfPoints": [1],
                "outputFields": ["primitiveVars", "Cp"],
                "type": "lineProbe",
            },
            "prb 122": {
                "animationFrequency": 1,
                "animationFrequencyOffset": 0,
                "computeTimeAverages": False,
                "outputFields": ["My_field_2"],
                "surfaces": ["surface21", "surface22"],
                "type": "surfaceIntegral",
            },
            "prb average": {
                "animationFrequency": 1,
                "animationFrequencyOffset": 0,
                "animationFrequencyTimeAverage": 10,
                "animationFrequencyTimeAverageOffset": 0,
                "startAverageIntegrationStep": -1,
                "computeTimeAverages": True,
                "start": [[10e-2, 10.02e-2, 10.03e-2]],
                "end": [[10e-2, 10.02e-2, 10.03e-2]],
                "numberOfPoints": [1],
                "outputFields": ["primitiveVars", "Cp", "T"],
                "type": "lineProbe",
            },
        },
        "outputFields": [],
    }
    assert sorted(ref.items()) == sorted(translated["monitorOutput"].items())


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
            ),
        ],
        {
            "observers": [[0.2, 0.02, 0.03], [0.0001, 0.02, 0.03]],
            "writePerSurfaceOutput": True,
            "patchType": "solid",
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
        },
    )


def test_acoustic_output(aeroacoustic_output_config, aeroacoustic_output_permeable_config):
    with SI_unit_system:
        param = SimulationParams(outputs=aeroacoustic_output_config[0])
    translated = {"boundaries": {}}
    param = param.preprocess(mesh_unit=1 * u.m, exclude=["models"])
    translated = translate_output(param, translated)

    assert sorted(aeroacoustic_output_config[1].items()) == sorted(
        translated["aeroacousticOutput"].items()
    )

    with SI_unit_system:
        param = SimulationParams(outputs=aeroacoustic_output_permeable_config[0])
    translated = {"boundaries": {}}
    param = param.preprocess(mesh_unit=1 * u.m, exclude=["models"])
    translated = translate_output(param, translated)

    assert sorted(aeroacoustic_output_permeable_config[1].items()) == sorted(
        translated["aeroacousticOutput"].items()
    )


def test_surface_slice_output():
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
                output_fields=["Cp", "Cf", "primitiveVars"],
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
                output_fields=["Mach", "primitiveVars", "yPlus"],
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
                    "outputFields": ["Cp", "Cf", "primitiveVars"],
                    "surfacePatches": ["zoneA/surface1", "zoneA/surface2"],
                    "animationFrequency": 1,
                    "animationFrequencyOffset": 0,
                    "computeTimeAverages": False,
                },
                {
                    "name": "S3",
                    "sliceOrigin": [0.01, 0.0101, 0.0003],
                    "sliceNormal": [0.0, 1.0, 0.0],
                    "outputFields": ["Cp", "Cf", "primitiveVars"],
                    "surfacePatches": ["zoneA/surface1", "zoneA/surface2"],
                    "animationFrequency": 1,
                    "animationFrequencyOffset": 0,
                    "computeTimeAverages": False,
                },
                {
                    "name": "P1",
                    "sliceOrigin": [0.01, 0.0102, 0.0003],
                    "sliceNormal": [0.0, 0.0, 1.0],
                    "outputFields": ["Mach", "primitiveVars", "yPlus"],
                    "surfacePatches": ["zoneB/surface1", "zoneB/surface2"],
                    "animationFrequency": 1,
                    "animationFrequencyOffset": 0,
                    "computeTimeAverages": False,
                },
                {
                    "name": "P2",
                    "sliceOrigin": [2.0, 1.01, 0.03],
                    "sliceNormal": [0.0, 0.0, -1.0],
                    "outputFields": ["Mach", "primitiveVars", "yPlus"],
                    "surfacePatches": ["zoneB/surface1", "zoneB/surface2"],
                    "animationFrequency": 1,
                    "animationFrequencyOffset": 0,
                    "computeTimeAverages": False,
                },
                {
                    "name": "P3",
                    "sliceOrigin": [3.0, 1.02, 0.03],
                    "sliceNormal": [0.0, 0.0, 1.0],
                    "outputFields": ["Mach", "primitiveVars", "yPlus"],
                    "surfacePatches": ["zoneB/surface1", "zoneB/surface2"],
                    "animationFrequency": 1,
                    "animationFrequencyOffset": 0,
                    "computeTimeAverages": False,
                },
            ],
        },
    )

    with SI_unit_system:
        param = SimulationParams(outputs=param_with_ref[0])
    param = param.preprocess(mesh_unit=1.0 * u.m, exclude=["models"])

    translated = {"boundaries": {}}
    translated = translate_output(param, translated)
    print(json.dumps(translated, indent=4))
    assert sorted(param_with_ref[1].items()) == sorted(translated["surfaceSliceOutput"].items())
