"""Tests that the stock mass-flow controller is skipped when a user UDD drives the patch."""

from types import SimpleNamespace

import flow360.component.simulation.units as u
from flow360.component.simulation.models.surface_models import (
    Inflow,
    MassFlowRate,
    Outflow,
)
from flow360.component.simulation.primitives import Surface
from flow360.component.simulation.translator.solver_translator import (
    mass_flow_default_udd,
)


def _mass_inflow(entity):
    return Inflow(
        entities=[entity],
        total_temperature=300 * u.K,
        spec=MassFlowRate(value=1.0 * u.kg / u.s),
    )


def _mass_outflow(entity):
    return Outflow(entities=[entity], spec=MassFlowRate(value=1.0 * u.kg / u.s))


def _user_controller(target, output_var):
    """Stand-in for a user UDD; the skip logic only reads output_target/output_vars."""
    return SimpleNamespace(
        name="userController",
        output_target=target,
        output_vars={output_var: "state[0];"},
    )


def _names(udds):
    return [udd.name for udd in (udds or [])]


def test_stock_controllers_generated_without_user_udd():
    inlet, outlet = Surface(name="fluid/inflow"), Surface(name="fluid/outflow")
    names = _names(mass_flow_default_udd([_mass_inflow(inlet), _mass_outflow(outlet)], None))
    assert any(n.startswith("massInflowController") for n in names)
    assert any(n.startswith("massOutflowController") for n in names)


def test_stock_inflow_controller_skipped_when_user_controls_patch():
    inlet, outlet = Surface(name="fluid/inflow"), Surface(name="fluid/outflow")
    user = _user_controller(inlet, "totalPressureRatio")
    result = mass_flow_default_udd([_mass_inflow(inlet), _mass_outflow(outlet)], [user])
    names = _names(result)
    assert user in result
    assert not any(n.startswith("massInflowController") for n in names)
    assert any(n.startswith("massOutflowController") for n in names)


def test_stock_outflow_controller_skipped_when_user_controls_patch():
    outlet = Surface(name="fluid/outflow")
    user = _user_controller(outlet, "staticPressureRatio")
    names = _names(mass_flow_default_udd([_mass_outflow(outlet)], [user]))
    assert not any(n.startswith("massOutflowController") for n in names)


def test_stock_controller_kept_when_user_writes_other_variable():
    inlet = Surface(name="fluid/inflow")
    user = _user_controller(inlet, "alphaAngle")
    names = _names(mass_flow_default_udd([_mass_inflow(inlet)], [user]))
    assert any(n.startswith("massInflowController") for n in names)
