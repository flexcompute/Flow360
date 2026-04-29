import importlib


def test_importable_artifact():
    module = importlib.import_module("flow360_schema")
    assert module.__name__ == "flow360_schema"
