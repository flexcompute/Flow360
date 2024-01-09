import json

import flow360 as fl


def test_schema_generators():
    types = [
        fl.NavierStokesSolver,
        fl.Geometry,
        fl.SlidingInterface,
        fl.SpalartAllmaras,
        fl.KOmegaSST,
        fl.TransitionModelSolver,
        fl.HeatEquationSolver,
        fl.NoneSolver,
        fl.PorousMedium,
        fl.ActuatorDisk,
        fl.BETDisk,
        fl.SurfaceOutput,
        fl.SliceOutput,
        fl.VolumeOutput,
        fl.AeroacousticOutput,
        fl.MonitorOutput,
        fl.IsoSurfaceOutput,
        fl.FreestreamFromVelocity,
        fl.FreestreamFromMach,
        fl.ZeroFreestreamFromVelocity,
        fl.FreestreamFromMachReynolds,
    ]

    for item in types:
        schema = item.flow360_schema()
        assert schema

        string = json.dumps(schema)
        assert string

        ui_schema = item.flow360_ui_schema()
        assert ui_schema

        string = json.dumps(ui_schema)
        assert string


def test_self_named_schema_generators():
    self_named_types = [
        fl.Surfaces,
        fl.VolumeZones,
        fl.Boundaries,
        fl.Slices,
        fl.IsoSurfaces,
        fl.Monitors,
    ]

    for item in self_named_types:
        schema = item.flow360_schema()
        assert schema

        string = json.dumps(schema)
        assert string

        ui_schema = item.flow360_ui_schema()
        assert ui_schema

        string = json.dumps(ui_schema)
        assert string
