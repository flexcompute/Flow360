import flow360 as fl
from flow360.component.simulation.framework.param_utils import AssetCache


def test_reference_geometry_fill_defaults_none():
    with fl.SI_unit_system:
        params = fl.SimulationParams(
            reference_geometry=None,
            operating_condition=fl.AerospaceCondition(velocity_magnitude=100 * fl.u.m / fl.u.s),
            private_attribute_asset_cache=AssetCache(project_length_unit=10 * fl.u.m),
        )
        filled = fl.ReferenceGeometry.fill_defaults(None, params)
        assert filled.area.to("m**2").value == 100
        assert all(c.to("m").value == 0.0 for c in filled.moment_center)
        assert all(l.to("m").value == 10.0 for l in filled.moment_length)


def test_reference_geometry_fill_defaults_partial():
    with fl.SI_unit_system:
        ref = fl.ReferenceGeometry(
            area=None,
            moment_center=(1, 2, 3) * fl.u.m,
            moment_length=None,
        )
        params = fl.SimulationParams(
            reference_geometry=ref,
            operating_condition=fl.AerospaceCondition(velocity_magnitude=100 * fl.u.m / fl.u.s),
            private_attribute_asset_cache=AssetCache(project_length_unit=1 * fl.u.m),
        )
        filled = fl.ReferenceGeometry.fill_defaults(ref, params)
        assert filled.area.to("m**2").value == 1.0
        assert [c.to("m").value for c in filled.moment_center] == [1.0, 2.0, 3.0]
        assert all(l.to("m").value == 1.0 for l in filled.moment_length)
