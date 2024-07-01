import json

import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.models.surface_models import Wall
from flow360.component.simulation.outputs.output_entities import Surface, SurfaceList
from flow360.component.simulation.outputs.outputs import (
    AeroAcousticOutput,
    Isosurface,
    IsosurfaceOutput,
    Probe,
    ProbeOutput,
    Slice,
    SliceOutput,
    SurfaceIntegralOutput,
    SurfaceList,
    SurfaceOutput,
    TimeAverageSurfaceOutput,
    TimeAverageVolumeOutput,
    VolumeOutput,
)
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
def surface_output_config_with_global_setting():
    return (
        [
            SurfaceOutput(  # Global
                frequency=11,
                frequency_offset=21,
                output_format="paraview",
                output_fields=["vorticity", "mutRatio"],
            ),
            SurfaceOutput(  # Local
                entities=[Surface(name="surface1"), Surface(name="surface2")],
                output_fields=["Cp"],
            ),
            SurfaceOutput(  # Local
                entities=[
                    Surface(name="surface11", private_attribute_full_name="ZoneName/surface11"),
                    Surface(name="surface22"),
                ],
                output_fields=["T"],
            ),
        ],
        {
            "animationFrequency": 11,
            "animationFrequencyOffset": 21,
            "animationFrequencyTimeAverage": -1,
            "animationFrequencyTimeAverageOffset": 0,
            "computeTimeAverages": False,
            "outputFields": [],
            "outputFormat": "paraview",
            "startAverageIntegrationStep": -1,
            "surfaces": {
                "surface1": {"outputFields": ["Cp", "vorticity", "mutRatio"]},
                "ZoneName/surface11": {"outputFields": ["T", "vorticity", "mutRatio"]},
                "surface2": {"outputFields": ["Cp", "vorticity", "mutRatio"]},
                "surface22": {"outputFields": ["T", "vorticity", "mutRatio"]},
                "Wall1": {"outputFields": ["vorticity", "mutRatio"]},
                "Wall2": {"outputFields": ["vorticity", "mutRatio"]},
            },
            "writeSingleFile": False,
        },
    )


@pytest.fixture()
def surface_output_config_with_no_global_setting():
    return (
        [
            SurfaceOutput(  # Local
                entities=[Surface(name="surface1"), Surface(name="surface2")],
                output_fields=["Cp"],
                output_format="tecplot",
            ),
            SurfaceOutput(  # Local
                entities=[Surface(name="surface11"), Surface(name="surface22")],
                frequency=123,
                frequency_offset=321,
                output_fields=["T"],
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
def avg_surface_output_config_with_global_setting():
    return [
        TimeAverageSurfaceOutput(  # Global
            frequency=23,
            frequency_offset=21,
            output_format="tecplot",
            output_fields=[],
            start_step=12,
            write_single_file=True,
        ),
        TimeAverageSurfaceOutput(  # Local
            entities=[Surface(name="surface1"), Surface(name="surface2")],
            output_fields=["Cp"],
        ),
        TimeAverageSurfaceOutput(  # Local
            entities=[Surface(name="surface11"), Surface(name="surface22")],
            output_fields=["T"],
        ),
    ]


def test_surface_ouput(
    surface_output_config_with_global_setting,
    surface_output_config_with_no_global_setting,
    avg_surface_output_config_with_global_setting,
):
    import json

    ##:: surfaceOutput with global settings
    with SI_unit_system:
        param = SimulationParams(outputs=surface_output_config_with_global_setting[0])
    translated = {
        "boundaries": {
            "Wall1": {"type": "NoSlipWall"},
            "Wall2": {"type": "NoSlipWall"},
            "surface1": {"type": "NoSlipWall"},
            "ZoneName/surface11": {"type": "NoSlipWall"},
            "surface2": {"type": "NoSlipWall"},
            "surface22": {"type": "NoSlipWall"},
        }
    }
    translated = translate_output(param, translated)
    assert sorted(surface_output_config_with_global_setting[1].items()) == sorted(
        translated["surfaceOutput"].items()
    )

    ##:: surfaceOutput with No global settings
    with SI_unit_system:
        param = SimulationParams(outputs=surface_output_config_with_no_global_setting[0])
    translated = {"boundaries": {}}
    translated = translate_output(param, translated)
    assert sorted(surface_output_config_with_no_global_setting[1].items()) == sorted(
        translated["surfaceOutput"].items()
    )

    ##:: timeAverageSurfaceOutput and surfaceOutput
    with SI_unit_system:
        param = SimulationParams(
            outputs=surface_output_config_with_no_global_setting[0]
            + avg_surface_output_config_with_global_setting
        )
    translated = {"boundaries": {}}
    translated = translate_output(param, translated)
    ref = {
        "animationFrequency": 123,
        "animationFrequencyOffset": 321,
        "animationFrequencyTimeAverage": 23,
        "animationFrequencyTimeAverageOffset": 21,
        "computeTimeAverages": True,
        "outputFields": [],
        "outputFormat": "tecplot",
        "startAverageIntegrationStep": 12,
        "surfaces": {
            "surface1": {"outputFields": ["Cp"]},
            "surface11": {"outputFields": ["T"]},
            "surface2": {"outputFields": ["Cp"]},
            "surface22": {"outputFields": ["T"]},
        },
        "writeSingleFile": True,
    }
    assert sorted(ref.items()) == sorted(translated["surfaceOutput"].items())


@pytest.fixture()
def sliceoutput_config_with_global_setting():
    return (
        [
            SliceOutput(  # Global
                frequency=33,
                frequency_offset=22,
                output_format="both",
                output_fields=["primitiveVars", "wallDistance"],
            ),
            SliceOutput(  # Local
                entities=[
                    Slice(
                        name="slice10",
                        normal=(0, 0, 2),
                        origin=(0.02, 0.03, 0.04) * u.m,
                    ),
                    Slice(
                        name="slice20",
                        normal=(0, 3, 4),
                        origin=(0.12, 0.13, 0.14) * u.m,
                    ),
                ],
                output_fields=["Cp"],
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
                        normal=(4, 0, 3),
                        origin=(6.12, 6.13, 6.14) * u.m,
                    ),
                ],
                output_fields=["T", "qcriterion"],
            ),
        ],
        {
            "animationFrequency": 33,
            "animationFrequencyOffset": 22,
            "outputFields": [],
            "outputFormat": "paraview,tecplot",
            "slices": {
                "slice01": {
                    "outputFields": ["T", "qcriterion", "primitiveVars", "wallDistance"],
                    "sliceNormal": [1.0, 0.0, 0.0],
                    "sliceOrigin": [10.02, 10.03, 10.04],
                },
                "slice02": {
                    "outputFields": ["T", "qcriterion", "primitiveVars", "wallDistance"],
                    "sliceNormal": [0.8, 0.0, 0.6],
                    "sliceOrigin": [6.12, 6.13, 6.14],
                },
                "slice10": {
                    "outputFields": ["Cp", "primitiveVars", "wallDistance"],
                    "sliceNormal": [0.0, 0.0, 1],
                    "sliceOrigin": [0.02, 0.03, 0.04],
                },
                "slice20": {
                    "outputFields": ["Cp", "primitiveVars", "wallDistance"],
                    "sliceNormal": [0.0, 0.6, 0.8],
                    "sliceOrigin": [0.12, 0.13, 0.14],
                },
            },
        },
    )


@pytest.fixture()
def sliceoutput_config_with_no_global_setting():
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


def test_slice_ouput(
    sliceoutput_config_with_global_setting,
    sliceoutput_config_with_no_global_setting,
):
    ##:: sliceOutput with global settings
    with SI_unit_system:
        param = SimulationParams(outputs=sliceoutput_config_with_global_setting[0])
    param = param.preprocess(1.0 * u.m, exclude=["models"])
    translated = {"boundaries": {}}
    translated = translate_output(param, translated)

    assert sorted(sliceoutput_config_with_global_setting[1].items()) == sorted(
        translated["sliceOutput"].items()
    )

    ##:: sliceOutput with NO global settings
    with SI_unit_system:
        param = SimulationParams(outputs=sliceoutput_config_with_no_global_setting[0])
    param = param.preprocess(1.0 * u.m, exclude=["models"])
    translated = {"boundaries": {}}
    translated = translate_output(param, translated)

    assert sorted(sliceoutput_config_with_no_global_setting[1].items()) == sorted(
        translated["sliceOutput"].items()
    )


@pytest.fixture()
def isosurface_output_config_with_global_setting():
    return (
        [
            IsosurfaceOutput(  # Global
                frequency=33,
                frequency_offset=22,
                output_format="both",
                output_fields=["primitiveVars", "wallDistance"],
            ),
            IsosurfaceOutput(  # Local
                entities=[
                    Isosurface(
                        name="isosurface 10",
                        iso_value=0.0001,
                        field="T",
                    ),
                    Isosurface(
                        name="isosurface 12",
                        iso_value=20.431,
                        field="vorticity",
                    ),
                ],
                output_fields=["Cp"],
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
                output_fields=["T", "primitiveVars"],
            ),
        ],
        {
            "animationFrequency": 33,
            "animationFrequencyOffset": 22,
            "isoSurfaces": {
                "isosurface 01": {
                    "outputFields": ["T", "primitiveVars", "wallDistance"],
                    "surfaceField": "nuHat",
                    "surfaceFieldMagnitude": 0.0001,
                },
                "isosurface 02": {
                    "outputFields": ["T", "primitiveVars", "wallDistance"],
                    "surfaceField": "qcriterion",
                    "surfaceFieldMagnitude": 10000.0,
                },
                "isosurface 10": {
                    "outputFields": ["Cp", "primitiveVars", "wallDistance"],
                    "surfaceField": "T",
                    "surfaceFieldMagnitude": 0.0001,
                },
                "isosurface 12": {
                    "outputFields": ["Cp", "primitiveVars", "wallDistance"],
                    "surfaceField": "vorticity",
                    "surfaceFieldMagnitude": 20.431,
                },
            },
            "outputFields": [],
            "outputFormat": "paraview,tecplot",
        },
    )


@pytest.fixture()
def isosurface_output_config_with_no_global_setting():
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
                        field="vorticity",
                    ),
                ],
                output_fields=["Cp"],
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
                    "surfaceField": "vorticity",
                    "surfaceFieldMagnitude": 20.431,
                },
            },
            "outputFields": [],
            "outputFormat": "paraview",
        },
    )


def test_isosurface_output(
    isosurface_output_config_with_global_setting,
    isosurface_output_config_with_no_global_setting,
):
    ##:: isoSurface with global settings
    with SI_unit_system:
        param = SimulationParams(outputs=isosurface_output_config_with_global_setting[0])
    translated = {"boundaries": {}}
    translated = translate_output(param, translated)
    assert sorted(isosurface_output_config_with_global_setting[1].items()) == sorted(
        translated["isoSurfaceOutput"].items()
    )

    ##:: isoSurface with NO global settings
    with SI_unit_system:
        param = SimulationParams(outputs=isosurface_output_config_with_no_global_setting[0])
    translated = {"boundaries": {}}
    translated = translate_output(param, translated)

    assert sorted(isosurface_output_config_with_no_global_setting[1].items()) == sorted(
        translated["isoSurfaceOutput"].items()
    )


@pytest.fixture()
def probe_output_config_with_global_setting():
    return (
        [
            ProbeOutput(  # Global
                output_fields=["primitiveVars", "T"],
            ),
            ProbeOutput(  # Local
                entities=[
                    Probe(
                        name="prb 10",
                        locations=[[1, 1.02, 0.03] * u.cm, [0.0001, 0.02, 0.03] * u.m],
                    ),
                    Probe(
                        name="prb 12",
                        locations=[[10, 10.02, 10.03] * u.cm],
                    ),
                ],
                output_fields=["primitiveVars", "Cp"],
            ),
        ],
        {
            "monitors": {
                "prb 10": {
                    "monitor_locations": [[1e-2, 1.02e-2, 0.0003], [0.0001, 0.02, 0.03]],
                    "outputFields": ["primitiveVars", "Cp", "T"],
                },
                "prb 12": {
                    "monitor_locations": [[10e-2, 10.02e-2, 10.03e-2]],
                    "outputFields": ["primitiveVars", "Cp", "T"],
                },
            },
            "outputFields": [],
        },
    )


@pytest.fixture()
def surface_integral_output_config_with_global_setting():
    return (
        [
            SurfaceIntegralOutput(  # Global
                output_fields=["primitiveVars", "T"],
            ),
            SurfaceIntegralOutput(  # Local
                entities=[
                    SurfaceList(
                        name="prb 110",
                        entities=[
                            Surface(
                                name="surface1", private_attribute_full_name="zoneName/surface1"
                            ),
                            Surface(name="surface2"),
                        ],
                    ),
                    SurfaceList(
                        name="prb 122",
                        entities=[Surface(name="surface21"), Surface(name="surface22")],
                    ),
                ],
                output_fields=["primitiveVars", "Cp", "T"],
            ),
        ],
        {
            "monitors": {
                "prb 110": {
                    "outputFields": ["primitiveVars", "Cp", "T"],
                    "surfaces": ["zoneName/surface1", "surface2"],
                },
                "prb 122": {
                    "outputFields": ["primitiveVars", "Cp", "T"],
                    "surfaces": ["surface21", "surface22"],
                },
            },
            "outputFields": [],
        },
    )


def test_monitor_output(
    probe_output_config_with_global_setting, surface_integral_output_config_with_global_setting
):
    ##:: monitorOutput with global probe settings
    with SI_unit_system:
        param = SimulationParams(outputs=probe_output_config_with_global_setting[0])
    param = param.preprocess(mesh_unit=1.0 * u.m, exclude=["models"])

    translated = {"boundaries": {}}
    translated = translate_output(param, translated)
    assert sorted(probe_output_config_with_global_setting[1].items()) == sorted(
        translated["monitorOutput"].items()
    )

    ##:: surfaceIntegral with global probe settings
    with SI_unit_system:
        param = SimulationParams(outputs=surface_integral_output_config_with_global_setting[0])
    param = param.preprocess(mesh_unit=1 * u.m, exclude=["models"])

    translated = {"boundaries": {}}
    translated = translate_output(param, translated)
    assert sorted(surface_integral_output_config_with_global_setting[1].items()) == sorted(
        translated["monitorOutput"].items()
    )

    ##:: surfaceIntegral and probeMonitor with global probe settings
    with SI_unit_system:
        param = SimulationParams(
            outputs=surface_integral_output_config_with_global_setting[0]
            + probe_output_config_with_global_setting[0]
        )
    param = param.preprocess(mesh_unit=1 * u.m, exclude=["models"])

    translated = {"boundaries": {}}
    translated = translate_output(param, translated)
    ref = {
        "monitors": {
            "prb 10": {
                "monitor_locations": [[1e-2, 1.02e-2, 0.0003], [0.0001, 0.02, 0.03]],
                "outputFields": ["primitiveVars", "Cp", "T"],
            },
            "prb 110": {
                "outputFields": ["primitiveVars", "Cp", "T"],
                "surfaces": ["zoneName/surface1", "surface2"],
            },
            "prb 12": {
                "monitor_locations": [[10e-2, 10.02e-2, 10.03e-2]],
                "outputFields": ["primitiveVars", "Cp", "T"],
            },
            "prb 122": {
                "outputFields": ["primitiveVars", "Cp", "T"],
                "surfaces": ["surface21", "surface22"],
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
                observers=[[0.2, 0.02, 0.03] * u.cm, [0.0001, 0.02, 0.03] * u.m],
                write_per_surface_output=True,
            ),
        ],
        {
            "observers": [[0.002, 0.0002, 0.0003], [0.0001, 0.02, 0.03]],
            "writePerSurfaceOutput": True,
            "patchType": "solid",
        },
    )


def test_acoustic_output(aeroacoustic_output_config):
    ##:: monitorOutput with global probe settings
    with SI_unit_system:
        param = SimulationParams(outputs=aeroacoustic_output_config[0])
    translated = {"boundaries": {}}
    param = param.preprocess(mesh_unit=1 * u.m, exclude=["models"])
    translated = translate_output(param, translated)

    assert sorted(aeroacoustic_output_config[1].items()) == sorted(
        translated["aeroacousticOutput"].items()
    )
