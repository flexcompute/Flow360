import pytest


@pytest.fixture(autouse=True)
def _clear_groups(geometry):
    """Clear face groups before each test so session-scoped geometry stays clean."""
    geometry.clear_groups()
    yield
