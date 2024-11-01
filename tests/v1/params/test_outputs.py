import unittest

import numpy as np
import pydantic.v1 as pd
import pytest
import unyt

import flow360.component.v1.units as u
import flow360.v1 as v1
from flow360.component.v1.flow360_output import (
    IsoSurface,
    IsoSurfaceOutput,
    MonitorOutput,
    ProbeMonitor,
    SliceOutput,
    Surface,
    SurfaceIntegralMonitor,
    SurfaceOutput,
    UserDefinedField,
    VolumeOutput,
)
from flow360.component.v1.flow360_params import (
    AeroacousticOutput,
    Flow360Params,
    FreestreamFromMach,
    Geometry,
)
from tests.utils import to_file_from_file_test

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

    output = SurfaceOutput(
        output_fields=["Cp", "qcriterion"],
    )

    assert output

    to_file_from_file_test(output)

    output = SurfaceOutput(
        output_fields=["Cp", "qcriterion"],
    )

    with v1.SI_unit_system:
        params = Flow360Params(
            surface_output=SurfaceOutput(
                output_fields=["Cp"],
                surfaces={"symmetry": Surface(output_fields=["Mach"])},
                output_format="both",
            ),
            boundaries={
                "1": v1.NoSlipWall(name="wing"),
                "2": v1.SlipWall(name="symmetry"),
                "3": v1.FreestreamBoundary(name="freestream"),
            },
            freestream=FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
        )
        solver_params = params.to_solver()

        assert solver_params.surface_output.output_format == "paraview,tecplot"
        assert "wing" in solver_params.surface_output.surfaces.names()
        assert "symmetry" in solver_params.surface_output.surfaces.names()
        assert "freestream" in solver_params.surface_output.surfaces.names()
        for surface_name, surface_item in solver_params.surface_output.surfaces.dict().items():
            if surface_name == "symmetry":
                assert {"Cp", "Mach"} == set(surface_item["output_fields"])
            else:
                assert surface_item["output_fields"] == ["Cp"]

    with v1.SI_unit_system:
        params = Flow360Params(
            surface_output=SurfaceOutput(
                output_fields=["Cp", "solutionTurbulence", "nuHat"],
                surfaces={"symmetry": Surface(output_fields=["Mach", "solutionTurbulence"])},
                output_format="tecplot",
            ),
            boundaries={
                "1": v1.NoSlipWall(name="wing"),
                "2": v1.SlipWall(name="symmetry"),
                "3": v1.FreestreamBoundary(name="freestream"),
            },
            freestream=FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
        )
        solver_params = params.to_solver()

        for surface_name, surface_item in solver_params.surface_output.surfaces.dict().items():
            if surface_name == "symmetry":
                assert {"Cp", "Mach", "solutionTurbulence"} == set(surface_item["output_fields"])
            else:
                assert set(surface_item["output_fields"]) == {"Cp", "solutionTurbulence"}


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
            output_fields=["Cp", "qcriterion"],
            slices={
                "sliceName_1": v1.Slice(
                    slice_normal=(0, 1, 0),
                    slice_origin=(0, 0.56413, 0) * u.m,
                ),
                "sliceName_2": v1.Slice(
                    slice_normal=(0, 0, 1),
                    slice_origin=(0, 0.56413 * u.inch, 0),
                    output_fields=["Mach"],
                ),
            },
        )

    output = SliceOutput(
        output_fields=["Cp", "qcriterion"],
        slices={
            "sliceName_1": v1.Slice(
                slice_normal=(0, 1, 0),
                slice_origin=(0, 0.56413, 0) * u.m,
            ),
            "sliceName_2": v1.Slice(
                slice_normal=(0, 0, 1),
                slice_origin=(0, 0.56413, 0) * u.inch,
                output_fields=["Mach"],
            ),
        },
    )

    assert output

    with pytest.raises(pd.ValidationError):
        output = SliceOutput(
            output_fields=["Cp", "qcriterion"],
            slices={
                "sliceName_1": v1.Slice(
                    slice_normal={0, 1},
                    slice_origin=(0, 0.56413, 0) * u.m,
                )
            },
        )

    with pytest.raises(unyt.exceptions.InvalidUnitOperation):
        output = SliceOutput(
            output_fields=["Cp", "qcriterion"],
            slices={
                "sliceName_1": v1.Slice(
                    slice_normal=(0, 1, 0),
                    slice_origin={0, 0.56413} * u.m,
                )
            },
        )

    output = SliceOutput(
        output_fields=["Cp", "qcriterion"],
        slices={
            "sliceName_1": v1.Slice(
                slice_normal=[0, 1, 0],
                slice_origin=(0, 0.56413, 0) * u.flow360_length_unit,
            ),
            "sliceName_2": v1.Slice(
                slice_normal=(0, 0, 1),
                slice_origin=(0, 0.56413, 0) * u.inch,
                output_fields=["Mach"],
            ),
        },
    )

    assert output

    to_file_from_file_test(output)

    output.output_format = "both"
    with v1.SI_unit_system:
        params = Flow360Params(
            slice_output=output,
            boundaries={},
            freestream=FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
            geometry=Geometry(mesh_unit=1),
        )
        solver_params = params.to_solver()

        assert solver_params.slice_output.output_format == "paraview,tecplot"

        for slice_name, slice_item in solver_params.slice_output.slices.dict().items():
            if slice_name == "sliceName_2":
                assert {"Cp", "Mach", "qcriterion"} == set(slice_item["output_fields"])
            else:
                assert {"Cp", "qcriterion"} == set(slice_item["output_fields"])

    with v1.SI_unit_system:
        params = Flow360Params(
            slice_output=SliceOutput(
                output_fields=[
                    "Cp",
                    "qcriterion",
                    "nuHat",
                    "solutionTurbulence",
                ],
                slices={
                    "sliceName_1": v1.Slice(
                        slice_normal=[5, 1, 0],
                        slice_origin=(0, 0.56413, 0) * u.flow360_length_unit,
                    ),
                    "sliceName_2": v1.Slice(
                        slice_normal=(0, 1, 1),
                        slice_origin=(0, 0.56413, 0) * u.inch,
                        output_fields=["Mach"],
                    ),
                },
            ),
            boundaries={},
            freestream=FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
            geometry=Geometry(mesh_unit=1),
        )
        solver_params = params.to_solver()

        for slice_name, slice_item in solver_params.slice_output.slices.dict().items():
            if slice_name == "sliceName_2":
                assert {"Cp", "Mach", "qcriterion", "solutionTurbulence"} == set(
                    slice_item["output_fields"]
                )
            else:
                assert {"Cp", "qcriterion", "solutionTurbulence"} == set(
                    slice_item["output_fields"]
                )
            assert (
                abs(
                    np.linalg.norm(
                        np.array(solver_params.slice_output.slices["sliceName_1"].slice_normal)
                    )
                    - 1
                )
                < 1e-10
            )
            assert (
                abs(
                    np.linalg.norm(
                        np.array(solver_params.slice_output.slices["sliceName_2"].slice_normal)
                    )
                    - 1
                )
                < 1e-10
            )


def test_volume_output():
    output = VolumeOutput()

    assert output

    with pytest.raises(pd.ValidationError):
        output = VolumeOutput(animation_frequency=-2, output_fields=["Cp", "qcriterion"])

    with pytest.raises(pd.ValidationError):
        output = VolumeOutput(animation_frequency=0, output_fields=["Cp", "qcriterion"])

    output = VolumeOutput(output_fields=["Cp", "qcriterion"])

    assert output

    output = VolumeOutput(
        animation_frequency_time_average=1,
        output_fields=["Cp", "qcriterion"],
    )

    assert output

    to_file_from_file_test(output)

    output = VolumeOutput(
        output_fields=["Cp", "qcriterion"],
    )

    output.output_format = "both"

    with v1.SI_unit_system:
        params = Flow360Params(
            volume_output=output,
            boundaries={},
            freestream=FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
        )
        solver_params = params.to_solver()

        assert set(solver_params.volume_output.output_fields) == {"Cp", "qcriterion"}

    with v1.SI_unit_system:
        """
        Test addition of betMetrics/betMetricsPerDisk from slice output field
        """
        params = Flow360Params(
            volume_output=output,
            slice_output=SliceOutput(
                output_fields=["betMetrics"],
                slices={
                    "sliceName_1": v1.Slice(
                        slice_normal=(0, 1, 0),
                        slice_origin=(0, 0.56413, 0) * u.m,
                    ),
                    "sliceName_2": v1.Slice(
                        slice_normal=(0, 1, 0),
                        slice_origin=(50, 0.56413, 0) * u.m,
                        output_fields=["betMetricsPerDisk"],
                    ),
                },
            ),
            boundaries={},
            freestream=FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
            geometry=Geometry(mesh_unit=1),
        )
        solver_params = params.to_solver()

        assert solver_params.volume_output.output_format == "paraview,tecplot"

    assert set(solver_params.volume_output.output_fields) == {
        "qcriterion",
        "Cp",
        "betMetrics",
        "betMetricsPerDisk",
    }

    output.output_fields = ["qcriterion", "Cp", "solutionTurbulence", "kOmega"]
    with v1.SI_unit_system:
        """
        Test removing duplicate output fields
        """
        params = Flow360Params(
            volume_output=output,
            boundaries={},
            freestream=FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
            geometry=Geometry(mesh_unit=1),
        )
        solver_params = params.to_solver()

    assert set(solver_params.volume_output.output_fields) == {
        "qcriterion",
        "Cp",
        "solutionTurbulence",
    }


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

    output.output_format = "both"

    with v1.SI_unit_system:
        params = Flow360Params(
            iso_surface_output=output,
            boundaries={},
            freestream=FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
        )
        solver_params = params.to_solver()

        assert solver_params.iso_surface_output.output_format == "paraview,tecplot"

        for (
            iso_surface_name,
            iso_surface_item,
        ) in solver_params.iso_surface_output.iso_surfaces.dict().items():
            if iso_surface_name == "s1":
                assert {"Cp", "Mach", "qcriterion"} == set(iso_surface_item["output_fields"])
            else:
                assert {"Mach"} == set(iso_surface_item["output_fields"])

    with v1.SI_unit_system:
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
                assert {"Cp", "Mach", "qcriterion", "solutionTurbulence"} == set(
                    iso_surface_item["output_fields"]
                )
            else:
                assert {"Mach", "solutionTurbulence"} == set(iso_surface_item["output_fields"])

    params = Flow360Params("../data/cases/case_udd.json")
    params_as_dict = params.flow360_dict()
    assert set(params_as_dict["isoSurfaceOutput"]["isoSurfaces"]["newKey"]["outputFields"]) == {
        "Mach",
        "Cp",
    }
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

    with v1.SI_unit_system:
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
                assert {"Cp", "Mach", "T"} == set(monitor_item["output_fields"])
            else:
                assert {"Cp", "qcriterion", "Mach"} == set(monitor_item["output_fields"])

    with v1.SI_unit_system:
        params = Flow360Params(
            monitor_output=MonitorOutput(
                output_fields=["Cp", "solutionTurbulence", "kOmega"],
                monitors={"m1": probe, "m2": integral},
            ),
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
                assert {"Cp", "solutionTurbulence", "T"} == set(monitor_item["output_fields"])
            else:
                assert {"Cp", "qcriterion", "solutionTurbulence"} == set(
                    monitor_item["output_fields"]
                )


def test_output_fields():
    with v1.SI_unit_system:
        params = Flow360Params(
            geometry=Geometry(mesh_unit=1),
            volume_output=VolumeOutput(output_fields=["Cp", "qcriterion", "my_var"]),
            surface_output=SurfaceOutput(output_fields=["primitiveVars", "my_var"]),
            slice_output=SliceOutput(
                output_fields=["primitiveVars", "my_var", "mutRatio"],
                slices={
                    "sliceName_1": v1.Slice(
                        slice_normal=[5, 1, 0], slice_origin=(0, 1.56413, 0), output_fields=["Mach"]
                    ),
                },
            ),
            iso_surface_output=IsoSurfaceOutput(
                output_fields=["qcriterion", "my_var"],
                iso_surfaces={
                    "s1": IsoSurface(
                        surface_field_magnitude=10.5,
                        surface_field="qcriterion",
                        output_fields=["Cp"],
                    ),
                },
            ),
            boundaries={},
            freestream=FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
            user_defined_fields=[UserDefinedField(name="my_var", expression="1+1;")],
        )

        assert set(params.volume_output.output_fields) == set(["Cp", "qcriterion", "my_var"])
        assert set(params.surface_output.output_fields) == set(["primitiveVars", "my_var"])
        params = params.to_solver()
        assert set(params.slice_output.slices["sliceName_1"].output_fields) == set(
            [
                "primitiveVars",
                "my_var",
                "mutRatio",
                "Mach",
            ]
        )
        assert set(params.iso_surface_output.iso_surfaces["s1"].output_fields) == set(
            [
                "qcriterion",
                "my_var",
                "Cp",
            ]
        )

    with pytest.raises(
        pd.ValidationError, match=r"surface_output:, prmitiveVars is not valid output field name."
    ):
        with v1.SI_unit_system:
            Flow360Params(
                surface_output=SurfaceOutput(output_fields=["prmitiveVars", "my_var"]),
                boundaries={},
                freestream=FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
                user_defined_fields=[UserDefinedField(name="my_var", expression="1+1;")],
            )

    with pytest.raises(
        pd.ValidationError, match=r"surface_output->wing:, my__var is not valid output field name."
    ):
        with v1.SI_unit_system:
            Flow360Params(
                surface_output=SurfaceOutput(
                    output_fields=["primitiveVars", "my__var"],
                    surfaces={"wing": Surface(output_fields=["Cf"])},
                ),
                boundaries={},
                freestream=FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
                user_defined_fields=[UserDefinedField(name="my_var", expression="1+1;")],
            )
