from flow360.component.v1.modules import Env


def test_version():
    Env.dev.active()
    print(Env.current)
    assert Env.current.name == "dev"
    Env.prod.active()
