import os
from unittest.mock import MagicMock

import flow360_schema.models.simulation.units as u
import pydantic as pd
import pytest
from flow360_schema.models.asset_cache import (
    AeroAcousticParentCaseCache,
    FrozenObserver,
)
from flow360_schema.models.simulation.outputs.outputs import (
    AeroAcousticOutput,
    Observer,
)
from flow360_schema.models.simulation.services import ValidationCalledBy, validate_model
from flow360_schema.models.simulation.simulation_params import SimulationParams
from flow360_schema.models.simulation.time_stepping.time_stepping import Unsteady
from flow360_schema.models.simulation.units import SI_unit_system
from flow360_schema.testing import dump_wire_legal

from flow360.component.geometry import Geometry, GeometryMeta
from flow360.component.project_utils import (
    fill_aero_acoustic_parent_case_cache,
    set_up_params_for_uploading,
)
from flow360.component.resource_base import local_metadata_builder


def _aeroacoustic_params(*, observer_time_step_size=None, step_size=1e-3, force_clean_start=False):
    with SI_unit_system:
        return SimulationParams(
            time_stepping=Unsteady(steps=10, step_size=step_size),
            outputs=[
                AeroAcousticOutput(
                    observers=[Observer(position=[1.0, 0.0, 1.75] * u.m, group_name="1")],
                    observer_time_step_size=observer_time_step_size,
                    force_clean_start=force_clean_start,
                )
            ],
        )


def _recorded(params):
    return params.private_attribute_asset_cache.aeroacoustic_parent_case


def test_submission_records_the_resolved_observer_time_step_size():
    """Unset means the CFD time step size, and the record states it rather than leaving it implicit."""
    params = _aeroacoustic_params(step_size=2e-3)

    fill_aero_acoustic_parent_case_cache(params)

    recorded = _recorded(params)
    assert recorded.recorded_observer_time_step_size == 2e-3 * u.s
    assert recorded.recorded_aeroacoustic_solver_start_time == 0 * u.s
    assert recorded.recorded_patch_type == "solid"
    assert len(recorded.recorded_observers) == 1


def test_submission_records_the_explicit_observer_time_step_size():
    params = _aeroacoustic_params(observer_time_step_size=5e-3, step_size=1e-3)

    fill_aero_acoustic_parent_case_cache(params)

    assert _recorded(params).recorded_observer_time_step_size == 5e-3 * u.s


def _parent_with_record(**kwargs):
    """A parent case's params, as its own submission recorded them."""
    parent_params = _aeroacoustic_params(**kwargs)
    fill_aero_acoustic_parent_case_cache(parent_params)
    return parent_params


def test_fork_inherits_the_record_from_the_parent_case():
    """The uploaded params replace the draft's copied document, so lineage comes from the
    actual parent -- even for mint params that never touched the parent's record."""
    parent_params = _parent_with_record(step_size=1e-3)
    params = _aeroacoustic_params(step_size=4e-3)

    fill_aero_acoustic_parent_case_cache(params, is_fork=True, parent_case_params=parent_params)

    recorded = _recorded(params)
    assert recorded.model_dump(mode="json") == _recorded(parent_params).model_dump(mode="json")
    assert recorded is not _recorded(parent_params)
    assert recorded.recorded_observer_time_step_size == 1e-3 * u.s


def test_fork_replaces_a_stale_record_from_an_unrelated_case():
    """Params seeded from case A then forked from case B must be held to B, not A."""
    unrelated_params = _parent_with_record(step_size=9e-3)
    parent_params = _parent_with_record(step_size=1e-3)
    params = _aeroacoustic_params(step_size=4e-3)
    # pylint: disable-next=protected-access
    params.private_attribute_asset_cache._force_set_attr(
        "aeroacoustic_parent_case", _recorded(unrelated_params)
    )

    fill_aero_acoustic_parent_case_cache(params, is_fork=True, parent_case_params=parent_params)

    assert _recorded(params).recorded_observer_time_step_size == 1e-3 * u.s


def test_fork_of_an_unreadable_parent_records_nothing():
    """Recording this fork's own settings would falsify the signal it continues, and the
    stale record in the params is not evidence either: absent means honestly unenforced."""
    params = _aeroacoustic_params(step_size=4e-3)
    fill_aero_acoustic_parent_case_cache(params)

    fill_aero_acoustic_parent_case_cache(params, is_fork=True, parent_case_params=None)

    assert _recorded(params) is None


def test_new_case_overwrites_a_record_it_was_copied_with():
    """Params reach a non-fork case by copy too; that case runs its own acoustic signal."""
    params = _aeroacoustic_params(step_size=4e-3)
    fill_aero_acoustic_parent_case_cache(params)
    # pylint: disable-next=protected-access
    _recorded(params)._force_set_attr("recorded_observer_time_step_size", 1e-3 * u.s)

    fill_aero_acoustic_parent_case_cache(params)

    assert _recorded(params).recorded_observer_time_step_size == 4e-3 * u.s


def test_fork_with_force_clean_start_records_its_own_settings():
    """A clean start restarts the aeroacoustic solver, so there is nothing to inherit."""
    parent_params = _parent_with_record(step_size=1e-3)
    params = _aeroacoustic_params(step_size=4e-3, force_clean_start=True)

    fill_aero_acoustic_parent_case_cache(params, is_fork=True, parent_case_params=parent_params)

    assert _recorded(params).recorded_observer_time_step_size == 4e-3 * u.s


def test_case_without_aeroacoustics_records_nothing():
    with SI_unit_system:
        params = SimulationParams(time_stepping=Unsteady(steps=10, step_size=1e-3))

    fill_aero_acoustic_parent_case_cache(params)

    assert _recorded(params) is None


def test_filled_document_passes_the_fork_immutability_check():
    """The record is filled from this very output and must survive the serialization
    round trip and the fork-immutability comparison unchanged."""
    params = _aeroacoustic_params(observer_time_step_size=2e-3)
    fill_aero_acoustic_parent_case_cache(params)

    _, errors, _ = validate_model(
        params_as_dict=dump_wire_legal(params),
        validated_by=ValidationCalledBy.SERVICE,
        root_item_type="VolumeMesh",
        validation_level=None,
    )

    assert errors is None


def test_new_case_from_a_child_case_records_its_own_settings():
    """`run_case` without `fork_from` creates a regular case, never an auto-fork -- even
    when the params were loaded from a child (forked) case. Such params carry the child's
    inherited record; leaking it into the new case would claim acoustic settings the new
    case does not run with (a clean start samples at its own CFD time step size)."""
    params = _aeroacoustic_params(step_size=1e-3)
    inherited = AeroAcousticParentCaseCache(
        recorded_patch_type="solid",
        recorded_observers=[FrozenObserver(position=[9.0, 9.0, 9.0] * u.m, group_name="9")],
        recorded_observer_time_step_size=2e-3 * u.s,
        recorded_aeroacoustic_solver_start_time=0.5 * u.s,
    )
    # pylint: disable=protected-access
    params.private_attribute_asset_cache._force_set_attr("aeroacoustic_parent_case", inherited)

    fill_aero_acoustic_parent_case_cache(params, is_fork=False)

    recorded = _recorded(params)
    assert recorded.recorded_observer_time_step_size == 1e-3 * u.s
    assert recorded.recorded_aeroacoustic_solver_start_time == 0 * u.s
    assert recorded.recorded_observers[0].group_name == "1"

    _, errors, _ = validate_model(
        params_as_dict=dump_wire_legal(params),
        validated_by=ValidationCalledBy.SERVICE,
        root_item_type="VolumeMesh",
        validation_level=None,
    )
    assert errors is None


def test_record_containers_are_immutable_and_do_not_alias_the_output():
    """Frozen fields stop rebinding but not in-place container mutation or aliasing:
    the record's sequences are tuples, and its observers are deep copies -- editing the
    params after the fill must not rewrite the record."""
    params = _aeroacoustic_params(step_size=1e-3)
    fill_aero_acoustic_parent_case_cache(params)
    recorded = _recorded(params)

    assert isinstance(recorded.recorded_observers, tuple)
    assert recorded.recorded_observers[0] is not params.outputs[0].observers[0]

    params.outputs[0].observers[0].group_name = "9"
    with pytest.raises(pd.ValidationError):
        recorded.recorded_observers[0].group_name = "9"

    assert recorded.recorded_observers[0].group_name == "1"


def _local_geometry_root_asset():
    geo = Geometry.from_local_storage(
        geometry_id="geo-ea3bb31e-2f85-4504-943c-7788d91c1ab0",
        local_storage_path=os.path.join(
            os.path.dirname(__file__), "data", "geometry_grouped_by_file"
        ),
        meta_data=GeometryMeta(
            **local_metadata_builder(
                id="geo-ea3bb31e-2f85-4504-943c-7788d91c1ab0",
                name="TEST",
                cloud_path_prefix="/",
                status="processed",
            )
        ),
    )
    geo.internal_registry = geo._entity_info.get_persistent_entity_registry(geo.internal_registry)
    return geo


@pytest.mark.parametrize("force_clean_start", [False, True])
def test_submission_workflow_records_then_forks(mock_response, force_clean_start):
    """The user's exact workflow, through set_up_params_for_uploading rather than the fill:
    submit a case (record populated from its own settings), then fork it with altered
    aeroacoustic settings -- the fork's record is inherited from the parent when the signal
    continues, and refreshed from its own settings under force_clean_start."""
    geo = _local_geometry_root_asset()

    parent_params = set_up_params_for_uploading(
        root_asset=geo,
        length_unit=1 * u.m,
        params=_aeroacoustic_params(observer_time_step_size=1e-3, step_size=1e-3),
        use_beta_mesher=False,
        use_geometry_AI=False,
    )
    parent_record = _recorded(parent_params)
    assert parent_record is not None
    assert parent_record.recorded_observer_time_step_size == 1e-3 * u.s

    parent_case = MagicMock()
    parent_case.params = parent_params

    fork_params = set_up_params_for_uploading(
        root_asset=geo,
        length_unit=1 * u.m,
        params=_aeroacoustic_params(step_size=4e-3, force_clean_start=force_clean_start),
        use_beta_mesher=False,
        use_geometry_AI=False,
        fork_from=parent_case,
    )

    recorded = _recorded(fork_params)
    if force_clean_start:
        assert recorded.recorded_observer_time_step_size == 4e-3 * u.s
    else:
        assert recorded.model_dump(mode="json") == parent_record.model_dump(mode="json")


def test_fork_of_a_parent_without_aeroacoustics_records_its_own_settings(mock_response, capsys):
    """The parent ran no aeroacoustics, so the signal starts with this fork: it records its
    own settings, and this perfectly legitimate lineage produces no warning."""
    geo = _local_geometry_root_asset()
    with SI_unit_system:
        quiet_parent = SimulationParams(time_stepping=Unsteady(steps=10, step_size=1e-3))
    parent_params = set_up_params_for_uploading(
        root_asset=geo,
        length_unit=1 * u.m,
        params=quiet_parent,
        use_beta_mesher=False,
        use_geometry_AI=False,
    )
    assert _recorded(parent_params) is None
    parent_case = MagicMock()
    parent_case.params = parent_params

    capsys.readouterr()
    fork_params = set_up_params_for_uploading(
        root_asset=geo,
        length_unit=1 * u.m,
        params=_aeroacoustic_params(step_size=4e-3),
        use_beta_mesher=False,
        use_geometry_AI=False,
        fork_from=parent_case,
    )

    assert _recorded(fork_params).recorded_observer_time_step_size == 4e-3 * u.s
    captured = capsys.readouterr()
    assert "aeroacoustic" not in (captured.out + captured.err).lower()


def test_fork_of_a_legacy_parent_with_aeroacoustics_records_nothing():
    """A parent that ran aeroacoustics but predates the record has unknown lineage: the
    fork may itself be continuing an inherited signal, so nothing is guessed."""
    legacy_parent_params = _aeroacoustic_params(step_size=1e-3)  # never filled: no record
    params = _aeroacoustic_params(step_size=4e-3)

    fill_aero_acoustic_parent_case_cache(
        params, is_fork=True, parent_case_params=legacy_parent_params
    )

    assert _recorded(params) is None


def test_fork_that_drops_aeroacoustics_clears_the_inherited_record():
    """A fork without an AeroAcousticOutput runs no acoustic solver: no signal continues,
    so the record is cleared rather than inherited -- a later fork that re-adds
    aeroacoustics starts a new signal and must not be held to the grandparent's."""
    parent_params = _parent_with_record(step_size=1e-3)
    with SI_unit_system:
        params = SimulationParams(time_stepping=Unsteady(steps=10, step_size=4e-3))

    fill_aero_acoustic_parent_case_cache(params, is_fork=True, parent_case_params=parent_params)

    assert _recorded(params) is None
