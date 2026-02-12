
import importlib


def test_importable_artifact():
    module = importlib.import_module("flow360_schemas")
    assert module.__name__ == "flow360_schemas"
