from flow360.v1 import Env


def test_version():
    Env.dev.active()
    print(Env.current)
    assert Env.current.name == "dev"
    Env.prod.active()
