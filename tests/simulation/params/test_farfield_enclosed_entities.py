"""Tests for farfield enclosed_entities validation across all farfield types."""

import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.meshing_param.meshing_specs import MeshingDefaults
from flow360.component.simulation.meshing_param.params import MeshingParams
from flow360.component.simulation.meshing_param.volume_params import (
    AutomatedFarfield,
    CustomZones,
    FullyMovingFloor,
    RotationVolume,
    UserDefinedFarfield,
    WindTunnelFarfield,
)
from flow360.component.simulation.primitives import (
    CustomVolume,
    Cylinder,
    Sphere,
    Surface,
)
from flow360.component.simulation.services import (
    ValidationCalledBy,
    clear_context,
    validate_model,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import SI_unit_system
from flow360.component.simulation.validation.validation_context import (
    ParamsValidationInfo,
)


@pytest.fixture(autouse=True)
def reset_context():
    clear_context()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FARFIELD_TYPES_ALL = [AutomatedFarfield, UserDefinedFarfield, WindTunnelFarfield]
FARFIELD_TYPES_REQUIRING_CUSTOM_ZONES = [UserDefinedFarfield, WindTunnelFarfield]


def _make_farfield(farfield_cls, **kwargs):
    """Instantiate a farfield zone with type-specific defaults."""
    if farfield_cls is WindTunnelFarfield:
        kwargs.setdefault("name", "wind tunnel")
        kwargs.setdefault("floor_type", FullyMovingFloor())
    return farfield_cls(**kwargs)


def _make_defaults(farfield_cls):
    """Return MeshingDefaults with type-specific extras (e.g. geometry_accuracy for WindTunnel)."""
    kwargs = {"boundary_layer_first_layer_thickness": 1e-4}
    if farfield_cls is WindTunnelFarfield:
        kwargs["geometry_accuracy"] = 1e-4
    return MeshingDefaults(**kwargs)


def _make_asset_cache(farfield_cls, *, use_inhouse_mesher=True):
    """Return AssetCache with type-specific extras (e.g. use_geometry_AI for WindTunnel)."""
    kwargs = {"use_inhouse_mesher": use_inhouse_mesher}
    if farfield_cls is WindTunnelFarfield:
        kwargs["use_geometry_AI"] = True
    return AssetCache(**kwargs)


def _validate(params):
    """Run validation and return (warnings, errors, info)."""
    return validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="SurfaceMesh",
        validation_level="VolumeMesh",
    )


def _make_custom_zones_with_volume():
    """Standard CustomZones fixture used across many tests."""
    return CustomZones(
        name="interior",
        entities=[
            CustomVolume(
                name="zone1",
                bounding_entities=[Surface(name="face1"), Surface(name="face2")],
            )
        ],
    )


def _make_rotor_disk():
    return Cylinder(
        name="rotor",
        center=(0, 0, 0) * u.m,
        axis=(0, 0, 1),
        height=1 * u.m,
        outer_radius=5 * u.m,
    )


# ---------------------------------------------------------------------------
# Group A: enclosed_entities + beta mesher = PASS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("farfield_cls", FARFIELD_TYPES_ALL, ids=lambda c: c.__name__)
def test_enclosed_entities_beta_mesher_positive(farfield_cls):
    """enclosed_entities with beta mesher should pass validation."""
    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=_make_defaults(farfield_cls),
                volume_zones=[
                    _make_custom_zones_with_volume(),
                    _make_farfield(
                        farfield_cls,
                        enclosed_entities=[Surface(name="face1"), Surface(name="face2")],
                    ),
                ],
            ),
            private_attribute_asset_cache=_make_asset_cache(farfield_cls),
        )
    _, errors, _ = _validate(params)
    assert errors is None


# ---------------------------------------------------------------------------
# Group B: enclosed_entities + legacy mesher = FAIL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("farfield_cls", FARFIELD_TYPES_ALL, ids=lambda c: c.__name__)
def test_enclosed_entities_beta_mesher_negative(farfield_cls):
    """enclosed_entities with legacy mesher should fail validation."""
    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=_make_defaults(farfield_cls),
                volume_zones=[
                    _make_farfield(farfield_cls, enclosed_entities=[Surface(name="face1")]),
                ],
            ),
            private_attribute_asset_cache=_make_asset_cache(farfield_cls, use_inhouse_mesher=False),
        )
    _, errors, _ = _validate(params)
    assert errors is not None
    assert any(
        "`enclosed_entities` is only supported with the beta mesher" in e["msg"] for e in errors
    )


# ---------------------------------------------------------------------------
# Group C: Cylinder without RotationVolume = FAIL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("farfield_cls", FARFIELD_TYPES_ALL, ids=lambda c: c.__name__)
def test_enclosed_entities_rotation_volume_association_negative(farfield_cls):
    """Cylinder in enclosed_entities without a RotationVolume should fail."""
    rotor_disk = _make_rotor_disk()
    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=_make_defaults(farfield_cls),
                volume_zones=[
                    CustomZones(
                        name="interior",
                        entities=[
                            CustomVolume(
                                name="zone1",
                                bounding_entities=[Surface(name="face1")],
                            )
                        ],
                    ),
                    _make_farfield(
                        farfield_cls,
                        enclosed_entities=[Surface(name="face1"), rotor_disk],
                    ),
                ],
            ),
            private_attribute_asset_cache=_make_asset_cache(farfield_cls),
        )
    _, errors, _ = _validate(params)
    assert errors is not None
    assert any(
        "`Cylinder` entity `rotor` in `enclosed_entities` must be associated with a `RotationVolume`"
        in e["msg"]
        for e in errors
    )


# ---------------------------------------------------------------------------
# Group D: Cylinder with RotationVolume = PASS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("farfield_cls", FARFIELD_TYPES_ALL, ids=lambda c: c.__name__)
def test_enclosed_entities_rotation_volume_association_positive(farfield_cls):
    """Cylinder in enclosed_entities that is also in a RotationVolume should pass."""
    rotor_disk = _make_rotor_disk()
    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=_make_defaults(farfield_cls),
                volume_zones=[
                    RotationVolume(
                        entities=[rotor_disk],
                        spacing_axial=0.5 * u.m,
                        spacing_radial=0.5 * u.m,
                        spacing_circumferential=0.3 * u.m,
                    ),
                    CustomZones(
                        name="interior",
                        entities=[
                            CustomVolume(
                                name="zone1",
                                bounding_entities=[Surface(name="face1")],
                            )
                        ],
                    ),
                    _make_farfield(
                        farfield_cls,
                        enclosed_entities=[Surface(name="face1"), rotor_disk],
                    ),
                ],
            ),
            private_attribute_asset_cache=_make_asset_cache(farfield_cls),
        )
    _, errors, _ = _validate(params)
    assert errors is None


# ---------------------------------------------------------------------------
# Group E: enclosed_entities without CustomZones = FAIL (UserDefined/WindTunnel only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "farfield_cls", FARFIELD_TYPES_REQUIRING_CUSTOM_ZONES, ids=lambda c: c.__name__
)
def test_enclosed_entities_requires_custom_zones(farfield_cls):
    """enclosed_entities without CustomZones should fail."""
    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=_make_defaults(farfield_cls),
                volume_zones=[
                    _make_farfield(farfield_cls, enclosed_entities=[Surface(name="face1")]),
                ],
            ),
            private_attribute_asset_cache=_make_asset_cache(farfield_cls),
        )
    _, errors, _ = _validate(params)
    assert errors is not None
    assert any("only allowed when `CustomVolume` entities are present" in e["msg"] for e in errors)


# ---------------------------------------------------------------------------
# AutomatedFarfield-specific tests (not parameterizable)
# ---------------------------------------------------------------------------


def test_enclosed_entities_none_with_legacy_mesher():
    """enclosed_entities=None with legacy mesher should pass (no error from this validator)."""
    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(boundary_layer_first_layer_thickness=1e-4),
                volume_zones=[AutomatedFarfield()],
            ),
            private_attribute_asset_cache=AssetCache(use_inhouse_mesher=False),
        )
    _, errors, _ = _validate(params)
    assert errors is None


def test_enclosed_entities_surfaces_only_no_rotation_volume_needed():
    """Only Surface entities in enclosed_entities should not require a RotationVolume."""
    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(boundary_layer_first_layer_thickness=1e-4),
                volume_zones=[
                    _make_custom_zones_with_volume(),
                    AutomatedFarfield(
                        enclosed_entities=[Surface(name="face1"), Surface(name="face2")],
                    ),
                ],
            ),
            private_attribute_asset_cache=AssetCache(use_inhouse_mesher=True),
        )
    _, errors, _ = _validate(params)
    assert errors is None


# ---------------------------------------------------------------------------
# CustomVolume bounding_entities + rotation association tests
# ---------------------------------------------------------------------------


def test_custom_volume_enclosed_entities_rotation_volume_association_positive():
    """CustomVolume with Cylinder in bounding_entities that is in a RotationVolume should pass."""
    rotor = _make_rotor_disk()
    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(boundary_layer_first_layer_thickness=1e-4),
                volume_zones=[
                    RotationVolume(
                        entities=[rotor],
                        spacing_axial=0.5 * u.m,
                        spacing_radial=0.5 * u.m,
                        spacing_circumferential=0.3 * u.m,
                    ),
                    CustomZones(
                        name="interior",
                        entities=[
                            CustomVolume(
                                name="zone1",
                                bounding_entities=[Surface(name="face1"), rotor],
                            )
                        ],
                    ),
                    AutomatedFarfield(
                        enclosed_entities=[Surface(name="face1"), rotor],
                    ),
                ],
            ),
            private_attribute_asset_cache=AssetCache(use_inhouse_mesher=True),
        )
    _, errors, _ = _validate(params)
    assert errors is None


def test_custom_volume_enclosed_entities_rotation_volume_association_negative():
    """CustomVolume with Cylinder in bounding_entities without a RotationVolume should fail."""
    rotor = _make_rotor_disk()
    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(boundary_layer_first_layer_thickness=1e-4),
                volume_zones=[
                    CustomZones(
                        name="interior",
                        entities=[
                            CustomVolume(
                                name="zone1",
                                bounding_entities=[Surface(name="face1"), rotor],
                            )
                        ],
                    ),
                    AutomatedFarfield(
                        enclosed_entities=[Surface(name="face1")],
                    ),
                ],
            ),
            private_attribute_asset_cache=AssetCache(use_inhouse_mesher=True),
        )
    _, errors, _ = _validate(params)
    assert errors is not None
    assert any("`Cylinder` entity `rotor` in `CustomVolume` `zone1`" in e["msg"] for e in errors)


def test_custom_volume_enclosed_entities_sphere_rotation_volume_negative():
    """CustomVolume with Sphere in bounding_entities without a RotationVolume should fail."""
    sph = Sphere(
        name="sph",
        center=(0, 0, 0) * u.m,
        radius=5 * u.m,
    )
    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(boundary_layer_first_layer_thickness=1e-4),
                volume_zones=[
                    CustomZones(
                        name="interior",
                        entities=[
                            CustomVolume(
                                name="zone1",
                                bounding_entities=[Surface(name="face1"), sph],
                            )
                        ],
                    ),
                    AutomatedFarfield(
                        enclosed_entities=[Surface(name="face1")],
                    ),
                ],
            ),
            private_attribute_asset_cache=AssetCache(use_inhouse_mesher=True),
        )
    _, errors, _ = _validate(params)
    assert errors is not None
    assert any("`Sphere` entity `sph` in `CustomVolume` `zone1`" in e["msg"] for e in errors)


def test_custom_volume_in_farfield_enclosed_entities_rotation_volume_negative():
    """CustomVolume inside farfield enclosed_entities with Cylinder not in RotationVolume should fail."""
    rotor = _make_rotor_disk()
    with SI_unit_system:
        cv = CustomVolume(
            name="zone1",
            bounding_entities=[Surface(name="face1"), rotor],
        )
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(boundary_layer_first_layer_thickness=1e-4),
                volume_zones=[
                    CustomZones(
                        name="interior",
                        entities=[
                            cv,
                            CustomVolume(
                                name="zone2",
                                bounding_entities=[Surface(name="face2")],
                            ),
                        ],
                    ),
                    AutomatedFarfield(
                        enclosed_entities=[Surface(name="face2"), cv],
                    ),
                ],
            ),
            private_attribute_asset_cache=AssetCache(use_inhouse_mesher=True),
        )
    _, errors, _ = _validate(params)
    assert errors is not None
    assert any("`Cylinder` entity `rotor` in `CustomVolume` `zone1`" in e["msg"] for e in errors)


# ---------------------------------------------------------------------------
# Farfield + CustomVolume intersection tests
# ---------------------------------------------------------------------------


def test_farfield_custom_volume_no_intersection_positive():
    """CustomVolume in farfield enclosed_entities with disjoint entities should pass."""
    with SI_unit_system:
        cv = CustomVolume(
            name="inner_zone",
            bounding_entities=[Surface(name="cv_face1"), Surface(name="cv_face2")],
        )
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(boundary_layer_first_layer_thickness=1e-4),
                volume_zones=[
                    CustomZones(name="interior", entities=[cv]),
                    AutomatedFarfield(
                        enclosed_entities=[Surface(name="outer_face"), cv],
                    ),
                ],
            ),
            private_attribute_asset_cache=AssetCache(use_inhouse_mesher=True),
        )
    _, errors, _ = _validate(params)
    assert errors is None


def test_farfield_custom_volume_no_intersection_negative():
    """CustomVolume in farfield enclosed_entities sharing a surface with a sibling should fail."""
    shared_face = Surface(name="shared")
    with SI_unit_system:
        cv = CustomVolume(
            name="inner_zone",
            bounding_entities=[shared_face, Surface(name="cv_only")],
        )
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(boundary_layer_first_layer_thickness=1e-4),
                volume_zones=[
                    CustomZones(name="interior", entities=[cv]),
                    AutomatedFarfield(
                        enclosed_entities=[shared_face, cv],
                    ),
                ],
            ),
            private_attribute_asset_cache=AssetCache(use_inhouse_mesher=True),
        )
    _, errors, _ = _validate(params)
    assert errors is not None
    assert any("shares bounding entities" in e["msg"] for e in errors)
    assert any("shared" in e["msg"] for e in errors)


# ---------------------------------------------------------------------------
# CustomZones without farfield zone
# ---------------------------------------------------------------------------


def test_custom_zones_without_farfield():
    """CustomZones only (no farfield zone) should pass and resolve farfield_method to user-defined."""
    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(boundary_layer_first_layer_thickness=1e-4),
                volume_zones=[
                    _make_custom_zones_with_volume(),
                ],
            ),
            private_attribute_asset_cache=AssetCache(use_inhouse_mesher=True),
        )
    _, errors, _ = _validate(params)
    assert errors is None

    farfield_method = ParamsValidationInfo._get_farfield_method_(params.model_dump(mode="json"))
    assert farfield_method == "user-defined"
    assert params.meshing.farfield_method == "user-defined"


@pytest.mark.parametrize(
    "farfield_cls",
    [UserDefinedFarfield, WindTunnelFarfield],
    ids=lambda c: c.__name__,
)
def test_farfield_with_custom_volumes_requires_enclosed_entities(farfield_cls):
    """Explicit farfield + CustomZones but no enclosed_entities should fail."""
    with SI_unit_system:
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=_make_defaults(farfield_cls),
                volume_zones=[
                    _make_custom_zones_with_volume(),
                    _make_farfield(farfield_cls),
                ],
            ),
            private_attribute_asset_cache=_make_asset_cache(farfield_cls),
        )
    _, errors, _ = _validate(params)
    assert errors is not None
    assert any("`enclosed_entities` for farfield must be specified" in e["msg"] for e in errors)
