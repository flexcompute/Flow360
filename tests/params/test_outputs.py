import unittest

import pydantic as pd
import pytest
import unyt

import flow360
import flow360.units as u
from flow360.component.flow360_params.flow360_output import (
    AnimationSettings,
    AnimationSettingsExtended,
    IsoSurface,
    IsoSurfaceOutput,
    MonitorOutput,
    ProbeMonitor,
    SliceOutput,
    Surface,
    SurfaceIntegralMonitor,
    SurfaceOutput,
    VolumeOutput,
)
from flow360.component.flow360_params.flow360_params import (
    AeroacousticOutput,
    Flow360Params,
    FreestreamFromMach,
    Geometry,
)
from tests.utils import array_equality_override, to_file_from_file_test

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_aeroacoustic_output():
    output = AeroacousticOutput(observers=[(0, 0, 0), (0, 1, 1)])

    to_file_from_file_test(output)

    assert output

    output = AeroacousticOutput(
        observers=[(0, 0, 0), (0, 1, 1)],
        patch_type="solid",
    )

    assert output

    to_file_from_file_test(output)

    output = AeroacousticOutput(observers=[])

    to_file_from_file_test(output)

    assert output

    with pytest.raises(pd.ValidationError):
        output = AeroacousticOutput()

    with pytest.raises(pd.ValidationError):
        output = AeroacousticOutput(observers=[(0, 0, 0), (0, 1, 1)], animation_frequency=0)

    with pytest.raises(pd.ValidationError):
        output = AeroacousticOutput(
            observers=[(0, 0, 0), (0, 1, 1)],
            patch_type="other",
        )


def test_surface_output():
    output = SurfaceOutput()

    assert output

    with pytest.raises(pd.ValidationError):
        output = SurfaceOutput(
            animation_frequency=-2,
            output_fields=["Cp", "qcriterion"],
        )

    with pytest.raises(pd.ValidationError):
        output = SurfaceOutput(
            animation_frequency_time_average=-2,
            output_fields=["Cp", "qcriterion"],
        )

    with pytest.raises(pd.ValidationError):
        output = SurfaceOutput(
            output_fields=["invalid_field", "qcriterion"],
        )

    output = SurfaceOutput(
        output_fields=["Cp", "qcriterion"],
    )

    assert output

    to_file_from_file_test(output)

    output = SurfaceOutput(
        output_fields=["Coefficient of pressure", "qcriterion"],
    )

    with flow360.SI_unit_system:
        params = Flow360Params(
            surface_output=SurfaceOutput(
                output_fields=["Cp"], surfaces={"symmetry": Surface(output_fields=["Mach"])}
            ),
            boundaries={
                "1": flow360.NoSlipWall(name="wing"),
                "2": flow360.SlipWall(name="symmetry"),
                "3": flow360.FreestreamBoundary(name="freestream"),
            },
            freestream=FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
        )
        solver_params = params.to_solver()
        assert "wing" in solver_params.surface_output.surfaces.names()
        assert "symmetry" in solver_params.surface_output.surfaces.names()
        assert "freestream" in solver_params.surface_output.surfaces.names()
        for surface_name, surface_item in solver_params.surface_output.surfaces.dict().items():
            if surface_name == "symmetry":
                assert set(["Cp", "Mach"]) == set(surface_item["output_fields"])
            else:
                assert surface_item["output_fields"] == ["Cp"]


@pytest.mark.usefixtures("array_equality_override")
def test_slice_output():
    with pytest.raises(pd.ValidationError):
        output = SliceOutput(animation_frequency=-2, output_fields=["Cp", "qcriterion"])

    with pytest.raises(pd.ValidationError):
        output = SliceOutput(
            animation_frequency_offset=0,
            output_fields=["invalid_field", "qcriterion"],
        )

    with pytest.raises(pd.ValidationError):
        output = SliceOutput(
            output_fields=["Coefficient of pressure", "qcriterion"],
            slices={
                "sliceName_1": flow360.Slice(
                    slice_normal=(0, 1, 0),
                    slice_origin=(0, 0.56413, 0) * u.m,
                ),
                "sliceName_2": flow360.Slice(
                    slice_normal=(0, 0, 1),
                    slice_origin=(0, 0.56413 * u.inch, 0),
                    output_fields=["Mach"],
                ),
            },
        )

    output = SliceOutput(
        output_fields=["Coefficient of pressure", "qcriterion"],
        slices={
            "sliceName_1": flow360.Slice(
                slice_normal=(0, 1, 0),
                slice_origin=(0, 0.56413, 0) * u.m,
            ),
            "sliceName_2": flow360.Slice(
                slice_normal=(0, 0, 1),
                slice_origin=(0, 0.56413, 0) * u.inch,
                output_fields=["Mach"],
            ),
        },
    )

    assert output

    with pytest.raises(pd.ValidationError):
        output = SliceOutput(
            output_fields=["Coefficient of pressure", "qcriterion"],
            slices={
                "sliceName_1": flow360.Slice(
                    slice_normal={0, 1, 0},
                    slice_origin=(0, 0.56413, 0) * u.m,
                )
            },
        )

    with pytest.raises(unyt.exceptions.InvalidUnitOperation):
        output = SliceOutput(
            output_fields=["Coefficient of pressure", "qcriterion"],
            slices={
                "sliceName_1": flow360.Slice(
                    slice_normal=(0, 1, 0),
                    slice_origin={0, 0.56413, 0} * u.m,
                )
            },
        )

    output = SliceOutput(
        output_fields=["Coefficient of pressure", "qcriterion"],
        slices={
            "sliceName_1": flow360.Slice(
                slice_normal=[0, 1, 0],
                slice_origin=(0, 0.56413, 0) * u.flow360_length_unit,
            ),
            "sliceName_2": flow360.Slice(
                slice_normal=(0, 0, 1),
                slice_origin=(0, 0.56413, 0) * u.inch,
                output_fields=["Mach"],
            ),
        },
    )

    assert output

    to_file_from_file_test(output)

    with flow360.SI_unit_system:
        params = Flow360Params(
            slice_output=output,
            boundaries={},
            freestream=FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
            geometry=Geometry(mesh_unit=1),
        )
        solver_params = params.to_solver()
        for slice_name, slice_item in solver_params.slice_output.slices.dict().items():
            if slice_name == "sliceName_2":
                assert set(["Cp", "Mach", "qcriterion"]) == set(slice_item["output_fields"])
            else:
                assert set(["Cp", "qcriterion"]) == set(slice_item["output_fields"])


def test_volume_output():
    output = VolumeOutput()

    assert output

    with pytest.raises(pd.ValidationError):
        output = VolumeOutput(animation_frequency=-2, output_fields=["Cp", "qcriterion"])

    with pytest.raises(pd.ValidationError):
        output = VolumeOutput(animation_frequency=0, output_fields=["Cp", "qcriterion"])

    with pytest.raises(pd.ValidationError):
        output = VolumeOutput(
            animation_frequency=1,
            output_fields=["invalid_field", "qcriterion"],
        )

    output = VolumeOutput(output_fields=["Cp", "qcriterion"])

    assert output

    output = VolumeOutput(
        animation_frequency_time_average=1,
        output_fields=["Cp", "qcriterion"],
    )

    assert output

    to_file_from_file_test(output)

    output = VolumeOutput(
        output_fields=["Coefficient of pressure", "qcriterion"],
    )

    with flow360.SI_unit_system:
        params = Flow360Params(
            volume_output=output,
            boundaries={},
            freestream=FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
        )
        solver_params = params.to_solver()

        assert solver_params.volume_output.output_fields == ["Cp", "qcriterion"]


def test_iso_surface_output():
    iso_surface = IsoSurface(
        surface_field_magnitude=0.5,
        surface_field="qcriterion",
        output_fields=["Cp", "qcriterion"],
    )

    with pytest.raises(pd.ValidationError):
        output = IsoSurfaceOutput(
            animation_frequency=0,
            iso_surfaces={"s1": iso_surface},
        )

    output = IsoSurfaceOutput(
        output_fields=["Mach"],
        iso_surfaces={
            "s1": iso_surface,
            "s2": IsoSurface(
                surface_field_magnitude=0.2,
                surface_field="Cp",
            ),
        },
    )

    assert output

    to_file_from_file_test(output)

    with flow360.SI_unit_system:
        params = Flow360Params(
            iso_surface_output=output,
            boundaries={},
            freestream=FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
        )
        solver_params = params.to_solver()
        for (
            iso_surface_name,
            iso_surface_item,
        ) in solver_params.iso_surface_output.iso_surfaces.dict().items():
            if iso_surface_name == "s1":
                assert set(["Cp", "Mach", "qcriterion"]) == set(iso_surface_item["output_fields"])
            else:
                assert set(["Mach"]) == set(iso_surface_item["output_fields"])

    with flow360.SI_unit_system:
        params = Flow360Params(
            iso_surface_output=IsoSurfaceOutput(
                output_fields=["Mach", "kOmega", "solutionTurbulence"],
                iso_surfaces={
                    "s1": iso_surface,
                    "s2": IsoSurface(
                        surface_field_magnitude=0.2,
                        surface_field="Cp",
                    ),
                },
            ),
            boundaries={},
            freestream=FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
        )
        solver_params = params.to_solver()

        for (
            iso_surface_name,
            iso_surface_item,
        ) in solver_params.iso_surface_output.iso_surfaces.dict().items():
            if iso_surface_name == "s1":
                assert set(["Cp", "Mach", "kOmega", "qcriterion", "solutionTurbulence"]) == set(
                    iso_surface_item["output_fields"]
                )
            else:
                assert set(["Mach", "kOmega", "solutionTurbulence"]) == set(
                    iso_surface_item["output_fields"]
                )

    params = Flow360Params("../data/cases/case_udd.json")
    params_as_dict = params.flow360_dict()
    assert set(params_as_dict["isoSurfaceOutput"]["isoSurfaces"]["newKey"]["outputFields"]) == set(
        ["Mach", "Cp"]
    )
    assert set(params_as_dict["isoSurfaceOutput"]["outputFields"]) == set()


def test_monitor_output():
    probe = ProbeMonitor(monitor_locations=[[0, 0, 0], [0, 10, 0.4]], output_fields=["Cp", "T"])

    assert probe

    integral = SurfaceIntegralMonitor(
        surfaces=["surf1", "surf2"], output_fields=["Cp", "qcriterion"]
    )

    assert integral

    output = MonitorOutput(output_fields=["Cp", "Mach"], monitors={"m1": probe, "m2": integral})

    assert output

    to_file_from_file_test(output)

    with flow360.SI_unit_system:
        params = Flow360Params(
            monitor_output=output,
            boundaries={},
            freestream=FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
            geometry=Geometry(mesh_unit=1),
        )
        solver_params = params.to_solver()

        for (
            monitor_name,
            monitor_item,
        ) in solver_params.monitor_output.monitors.dict().items():
            if monitor_name == "m1":
                assert set(["Cp", "Mach", "T"]) == set(monitor_item["output_fields"])
            else:
                assert set(["Cp", "qcriterion", "Mach"]) == set(monitor_item["output_fields"])
