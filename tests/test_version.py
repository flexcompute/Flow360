from flow360.version import Flow360Version


def test_flow360version():
    assert Flow360Version("beta-0.3.0").tail == [0, 3, 0]
    assert Flow360Version("be--ta-0.3.1").tail == [0, 3, 1]
    assert Flow360Version("release-22.1.3.0").tail == [22, 1, 3, 0]
    assert Flow360Version("dummy-22.1.3.0").tail == [22, 1, 3, 0]
    assert Flow360Version("du-1m-m2y3-22.1.3.0").tail == [22, 1, 3, 0]
    assert Flow360Version("du-1m-m2y3-22.1.3.0").head == "du-1m-m2y3"

    assert Flow360Version("beta-21.4.999.999") < Flow360Version("beta-22.1.3.0")
    assert Flow360Version("release-21.4.999.999") < Flow360Version("beta-22.1.3.0")
    assert Flow360Version("release-20.4.999.1") < Flow360Version("beta-22.1.3.0")
    assert Flow360Version("release-0.3.0") < Flow360Version("beta-20.4.1.0")
    assert Flow360Version("release-0.3.0.999") < Flow360Version("release-0.3.1")
    assert Flow360Version("release-22.1.1.0") < Flow360Version("beta-22.1.1.1")
    assert Flow360Version("release-22.3.4.0") < Flow360Version("beta-22.3.4.100")
    assert Flow360Version("release-22.3.4.0") < Flow360Version("beta-22.3.10")
    assert Flow360Version("release-22.3.4.0") < Flow360Version("beta-22.20.10")
    assert Flow360Version("release-22.3.4.0") < Flow360Version("beta-100.20.10")

    assert not Flow360Version("release-0.3.1") < Flow360Version("release-0.3.1")
    assert not Flow360Version("release-22.1.3.0") < Flow360Version("release-22.1.3.0")
    assert not Flow360Version("release-22.2.1.0") < Flow360Version("release-0.3.1")

    assert Flow360Version("release-22.2.1.0") >= Flow360Version("release-22.2.1.0")
    assert Flow360Version("beta-22.10.1.1") >= Flow360Version("release-22.3.1.3")
    assert Flow360Version("release-22.2.1.0") == Flow360Version("release-22.2.1.0")
    assert Flow360Version("beta-22.2.1.0") != Flow360Version("release-22.2.1.0")
    assert Flow360Version("release-22.2.1.1") != Flow360Version("release-22.2.1.0")
