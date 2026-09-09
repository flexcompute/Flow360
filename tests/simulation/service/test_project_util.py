import flow360_schema.models.simulation.units as u
import numpy as np
import pytest
from flow360_schema.models.entities.surface_entities import GhostSphere
from flow360_schema.models.simulation.simulation_params import SimulationParams

from flow360.component.project_utils import _set_up_default_reference_geometry


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_legacy_ghost_surface_resolves_via_updater():
    # The updater converts the legacy GhostSurface token to a GhostSphere
    # reference, which the loader resolves to the full entity_info instance —
    # no client-side replacement pass needed anymore.
    params = SimulationParams.from_file("./data/simulation_with_old_ghost_surface.json")
    entity = params.models[1].entities.stored_entities[0]
    assert isinstance(entity, GhostSphere)  # This used to be GhostSurface by python
    assert entity.center == [5.0007498695, 0.0, 0.0]
    assert entity.max_radius == 504.16453591327473


def test_set_up_default_reference_geometry():
    params = SimulationParams.from_file("./data/simulation_with_old_ghost_surface.json")
    length_unit = 1 * u.cm
    new_params = _set_up_default_reference_geometry(params, length_unit=length_unit)

    assert np.all(new_params.reference_geometry.area == 1.0 * length_unit**2)
    assert np.all(new_params.reference_geometry.moment_center == (0.0, 0.0, 0.0) * length_unit)
