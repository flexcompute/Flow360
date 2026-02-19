from flow360.component.simulation.meshing_param.meshing_specs import MeshingDefaults
from flow360.component.simulation.validation.validation_context import (
    CASE,
    ValidationContext,
)


def test_meshing_defaults_removes_deprecated_remove_non_manifold_faces():
    payload = {"remove_non_manifold_faces": False}

    with ValidationContext(levels=CASE) as validation_context:
        defaults = MeshingDefaults.model_validate(payload)

    dumped = defaults.model_dump(mode="json", by_alias=True)
    assert "remove_non_manifold_faces" not in dumped

    warning_messages = [warning["msg"] for warning in validation_context.validation_warnings]
    assert len(warning_messages) == 1
    assert "remove_non_manifold_faces" in warning_messages[0]
