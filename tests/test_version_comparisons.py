import pytest

from flow360.exceptions import Flow360RuntimeError
from flow360.solver_version import Flow360Version, warn_if_minor_release


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


def test_flow360version_master_outranks_every_release():
    # a master build numbers itself independently of the release years, and is scaled up so
    # that it still compares as the newest version - example assets are picked that way
    assert Flow360Version("master-1.1.1.1").tail == [100, 100, 100, 100]
    assert Flow360Version("master-1.1.1.1") > Flow360Version("release-25.11")


def test_flow360version_minor_release():
    assert Flow360Version("release-25.8.7").is_minor_release
    assert Flow360Version("release-25.8.7").major_release == "release-25.8"

    # the major release itself
    assert not Flow360Version("release-25.8").is_minor_release
    # the scheme predating release-<year>.<month> has no major release to point at
    assert not Flow360Version("release-22.1.3.0").is_minor_release
    # non-release builds are internal by nature, no major release exists
    assert not Flow360Version("beta-25.8.7").is_minor_release
    assert not Flow360Version("master-25.8.7").is_minor_release


def test_warn_if_minor_release(capsys):
    capsys.readouterr()

    warn_if_minor_release("release-25.8.7")
    captured_text = " ".join(capsys.readouterr().out.split())
    assert (
        "You are running release-25.8.7, which is a minor solver release intended for internal "
        "testing only and is not recommended for general use. Please use the major release "
        "release-25.8 instead." in captured_text
    )

    warn_if_minor_release("release-25.8")
    assert capsys.readouterr().out == ""

    # nothing pinned, the version is inherited
    warn_if_minor_release(None)
    assert capsys.readouterr().out == ""


def test_warn_if_minor_release_rejects_invalid_version():
    with pytest.raises(Flow360RuntimeError, match="solver version is not valid: release-test"):
        warn_if_minor_release("release-test")


def test_flow360version_series():
    assert Flow360Version("release-25.11").series == Flow360Version("release-25.11.3").series
    assert Flow360Version("release-22.1.3.0").series == Flow360Version("release-22.1.4.0").series
    assert Flow360Version("release-25.11").series != Flow360Version("release-25.10").series
    assert Flow360Version("release-25.11").series != Flow360Version("beta-25.11").series

    # the key is hashable, so it can be used for grouping
    assert {Flow360Version("release-25.11").series, Flow360Version("release-25.11.3").series} == {
        ("release", 25, 11)
    }
