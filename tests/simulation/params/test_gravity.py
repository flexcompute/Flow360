import math

import flow360.component.simulation.units as u
from flow360.component.simulation.models.volume_models import Gravity
from flow360.component.simulation.translator.solver_translator import gravity_translator


def test_gravity_translator_default_direction():
    nondim_magnitude = 8.49e-5
    gravity = Gravity(
        direction=(0, 0, -1),
        magnitude=nondim_magnitude * u.m / u.s**2,
    )

    result = gravity_translator(gravity)

    assert "gravityVector" in result
    assert len(result["gravityVector"]) == 3
    assert math.isclose(result["gravityVector"][0], 0.0, abs_tol=1e-15)
    assert math.isclose(result["gravityVector"][1], 0.0, abs_tol=1e-15)
    assert math.isclose(result["gravityVector"][2], -nondim_magnitude, rel_tol=1e-10)


def test_gravity_translator_custom_direction():
    nondim_magnitude = 1e-3
    gravity = Gravity(
        direction=(1, 0, 0),
        magnitude=nondim_magnitude * u.m / u.s**2,
    )

    result = gravity_translator(gravity)

    assert "gravityVector" in result
    assert math.isclose(result["gravityVector"][0], nondim_magnitude, rel_tol=1e-10)
    assert math.isclose(result["gravityVector"][1], 0.0, abs_tol=1e-15)
    assert math.isclose(result["gravityVector"][2], 0.0, abs_tol=1e-15)
