import unittest

import pytest

import flow360.component.v1.modules as fl
from flow360.component.v1.boundaries import NoSlipWall
from flow360.component.v1.flow360_params import Flow360Params

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_duplidated_boundary_names():
    with fl.SI_unit_system:
        with pytest.raises(
            ValueError,
            match="Boundary name <wing> under patch <fluid/.*Wing> appears multiple times",
        ):
            Flow360Params(
                boundaries={
                    "fluid/fuselage": NoSlipWall(name="fuselage"),
                    "fluid/leftWing": NoSlipWall(name="wing"),
                    "fluid/rightWing": NoSlipWall(name="wing"),
                },
                freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
            )


def test_tri_quad_boundaries():
    """
    todo: handle warning
    """
    with fl.SI_unit_system:
        Flow360Params(
            boundaries={
                "fluid/tri_fuselage": NoSlipWall(),
                "fluid/quad_fuselage": NoSlipWall(),
                "fluid/tri_wing": NoSlipWall(),
                "fluid/quad_wing": NoSlipWall(),
            },
            freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
        )
