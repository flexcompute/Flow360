from flow360.python_client_version import Flow360ClientVersion


def test_Flow360ClientVersion():
    assert Flow360ClientVersion("v0.2.0b9").tail == 20
    assert Flow360ClientVersion("v0.2.0").tail == 20
    assert Flow360ClientVersion("v0.2.0000").tail == 20
    assert Flow360ClientVersion("v0.1.10").tail == 110
    assert Flow360ClientVersion("v7.9.3b2").tail == 793
    assert Flow360ClientVersion("v7.9.3b999").tail == 793

    assert Flow360ClientVersion(
        "v1.9.9") < Flow360ClientVersion("v2.0.0")
    assert Flow360ClientVersion(
        "v1.2.9") < Flow360ClientVersion("v1.3.0")
    assert Flow360ClientVersion(
        "v2.3.4") < Flow360ClientVersion("v2.3.5")
    assert Flow360ClientVersion(
        "v1.2.3b1") > Flow360ClientVersion("v1.1.3b9")

    assert Flow360ClientVersion(
        "v1.2.3b1") == Flow360ClientVersion("v1.2.3")
    assert Flow360ClientVersion(
        "v1.2.3b12") == Flow360ClientVersion("v1.2.3b3")
    assert Flow360ClientVersion(
        "v1.2.3b12") >= Flow360ClientVersion("v1.2.3b3")
    assert Flow360ClientVersion(
        "v1.2.3b1") >= Flow360ClientVersion("v1.1.3b9")
