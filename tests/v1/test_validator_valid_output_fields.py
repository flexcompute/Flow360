import unittest

import pytest

import flow360 as fl
import flow360.component.v1.units as u
from flow360.component.v1.boundaries import FreestreamBoundary, NoSlipWall
from flow360.component.v1.flow360_output import (
    IsoSurface,
    IsoSurfaceOutput,
    MonitorOutput,
    ProbeMonitor,
    Slice,
    SliceOutput,
    Surface,
    SurfaceIntegralMonitor,
    SurfaceOutput,
    VolumeOutput,
)
from flow360.component.v1.flow360_params import Flow360Params
from flow360.component.v1.solvers import NavierStokesSolver

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_numerical_dissipation_output_criterion():
    with fl.SI_unit_system:
        output_database = {
            "volume_output": VolumeOutput(
                output_fields=["T", "numericalDissipationFactor", "Mach"]
            ),
            "slice_output": SliceOutput(
                output_fields=["T"],
                slices={
                    "s1": Slice(
                        output_fields=["numericalDissipationFactor"],
                        slice_normal=(0, 1, 0),
                        slice_origin=(0, 0, 0),
                    ),
                    "s2": Slice(
                        output_fields=["Mach"],
                        slice_normal=(0, 1, 0),
                        slice_origin=(0, 1, 0),
                    ),
                    "s3": Slice(
                        slice_normal=(0, 1, 0),
                        slice_origin=(0, 1, 1) * u.cm,
                    ),
                },
            ),
            "surface_output": SurfaceOutput(
                output_fields=["T"],
                surfaces={
                    "s1": Surface(output_fields=["numericalDissipationFactor"]),
                    "s2": Surface(output_fields=["Mach"]),
                },
            ),
            "iso_surface_output": IsoSurfaceOutput(
                iso_surfaces={
                    "s1": IsoSurface(
                        output_fields=["numericalDissipationFactor", "T"],
                        surface_field="qcriterion",
                        surface_field_magnitude=12,
                    ),
                    "s2": IsoSurface(
                        outputFields=["Mach", "T"],
                        surface_field="qcriterion",
                        surface_field_magnitude=123,
                    ),
                },
            ),
            "monitor_output": MonitorOutput(
                output_fields=["T"],
                monitors={
                    "s1": ProbeMonitor(
                        output_fields=["numericalDissipationFactor"],
                        monitor_locations=[(0, 1, 2), (3, 4, 5)],
                    ),
                    "s2": SurfaceIntegralMonitor(outputFields=["Mach"], surfaces=["s1", "s2"]),
                },
            ),
        }

        param = Flow360Params(
            boundaries={"s1": NoSlipWall(), "s2": FreestreamBoundary()},
            surface_output=None,
            volume_output=None,
            slice_output=None,
            iso_surface_output=None,
            monitor_output=None,
            freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
            geometry=fl.Geometry(mesh_unit=1),
        )
        param.navier_stokes_solver = NavierStokesSolver(numerical_dissipation_factor=0.99)
        assert param
        for attr_name, attr_obj in output_database.items():
            with pytest.raises(
                ValueError,
                match="Numerical dissipation factor output requested, but low dissipation mode is not enabled",
            ):
                setattr(param, attr_name, attr_obj)

        param.navier_stokes_solver = NavierStokesSolver(numerical_dissipation_factor=0.2)
        for attr_name, attr_obj in output_database.items():
            setattr(param, attr_name, attr_obj)
        param.flow360_json()
