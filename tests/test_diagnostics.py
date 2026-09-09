import socket
from types import SimpleNamespace

import pytest

from flow360 import diagnostics
from flow360.diagnostics import (
    CheckResult,
    CheckStatus,
    DiagnosticReport,
    _check_api_key,
    _check_clock,
    _check_connectivity,
    _check_environment,
    _check_http,
    _check_install,
    _check_portal_auth,
    _check_project_list,
    _check_proxy,
    _check_storage,
    _check_tls,
    _check_webapi_auth,
    _CheckRunner,
    _render_report,
    _run_service_checks,
    _run_transport_checks,
    redact,
)
from flow360.environment import EnvironmentConfig
from flow360.exceptions import (
    Flow360AuthorisationError,
    Flow360WebError,
    Flow360WebNotFoundError,
)

PROXY_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)


@pytest.fixture(autouse=True)
def _clean_proxy_environment(monkeypatch):
    for name in PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def _local_env(port, scheme="http"):
    base = f"{scheme}://127.0.0.1:{port}"
    return EnvironmentConfig(
        name="local",
        domain="N/A",
        web_api_endpoint=f"{base}/flow360-api",
        web_url=f"{base}/flow360",
        portal_web_api_endpoint=f"{base}/flow360-portal-api",
    )


class _StubResponse:
    def __init__(self, status_code=200, headers=None, text=""):
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text


def test_redact():
    assert redact("short") == "****"
    assert redact("0123456789ab") == "****"
    assert redact("abcd-middle-part-wxyz") == "abcd...wxyz"


def test_report_ok_property():
    report = DiagnosticReport(
        checks=[
            CheckResult(name="a", status=CheckStatus.OK),
            CheckResult(name="b", status=CheckStatus.WARN),
            CheckResult(name="c", status=CheckStatus.SKIP),
        ]
    )
    assert report.ok

    report.checks.append(CheckResult(name="d", status=CheckStatus.FAIL))
    assert not report.ok


def test_runner_skips_dependents_of_failed_check():
    runner = _CheckRunner()
    runner.run("first", lambda: CheckResult(name="first", status=CheckStatus.FAIL))
    result = runner.run(
        "second", lambda: CheckResult(name="second", status=CheckStatus.OK), needs=("first",)
    )
    assert result.status is CheckStatus.SKIP
    assert "blocked by [first]" in result.detail

    chained = runner.run(
        "third", lambda: CheckResult(name="third", status=CheckStatus.OK), needs=("second",)
    )
    assert chained.status is CheckStatus.SKIP


def test_runner_warn_does_not_block_dependents():
    runner = _CheckRunner()
    runner.run("first", lambda: CheckResult(name="first", status=CheckStatus.WARN))
    result = runner.run(
        "second", lambda: CheckResult(name="second", status=CheckStatus.OK), needs=("first",)
    )
    assert result.status is CheckStatus.OK


def test_runner_converts_unexpected_exception_to_fail():
    runner = _CheckRunner()

    def broken():
        raise RuntimeError("boom")

    result = runner.run("broken", broken)
    assert result.status is CheckStatus.FAIL
    assert "RuntimeError: boom" in result.detail


def test_check_install_reports_versions():
    result = _check_install()
    assert result.status is CheckStatus.OK
    assert "flow360" in result.detail
    assert "Python" in result.detail


def test_check_environment_accepts_dedicated_portal_host():
    env = EnvironmentConfig.from_domain(name="prod", domain="simulation.cloud")
    result = _check_environment(env, "built-in prod")
    assert result.status is CheckStatus.OK
    assert "portal_web_api_endpoint = https://portal-api.simulation.cloud" in result.detail


def test_check_environment_warns_on_portal_equal_to_web_origin():
    env = EnvironmentConfig(
        name="onprem",
        domain="N/A",
        web_api_endpoint="http://localhost:80/flow360-api",
        web_url="http://localhost:80",
        portal_web_api_endpoint="http://localhost:80",
    )
    result = _check_environment(env, "in-memory")
    assert result.status is CheckStatus.WARN
    assert result.detail.splitlines()[0].startswith(
        "portal_web_api_endpoint looks like the web UI origin"
    )
    assert "from_on_premises_url" in result.fix


def test_check_environment_warns_on_missing_portal():
    env = EnvironmentConfig(
        name="onprem",
        domain="N/A",
        web_api_endpoint="http://localhost/flow360-api",
        web_url="http://localhost/flow360",
    )
    result = _check_environment(env, "in-memory")
    assert result.status is CheckStatus.WARN
    assert result.detail.splitlines()[0].startswith("portal_web_api_endpoint is not set")
    assert "from_on_premises_url" in result.fix


def test_check_environment_accepts_derived_on_premises_env():
    env = EnvironmentConfig.from_on_premises_url(name="onprem", base_url="http://localhost:80")
    result = _check_environment(env, "in-memory")
    assert result.status is CheckStatus.OK


def test_check_proxy_ok_without_proxy_variables():
    result = _check_proxy(_local_env(80))
    assert result.status is CheckStatus.OK
    assert "no proxy environment variables set" in result.detail


def test_check_proxy_warns_when_endpoints_are_proxied(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.corp:3128")
    result = _check_proxy(_local_env(80))
    assert result.status is CheckStatus.WARN
    assert "web_api_endpoint" in result.detail.splitlines()[0]
    assert "NO_PROXY" in result.fix


def test_check_proxy_ok_when_no_proxy_covers_endpoints(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.corp:3128")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    result = _check_proxy(_local_env(80))
    assert result.status is CheckStatus.OK
    assert result.detail.splitlines()[0].endswith("all configured endpoints bypass them")


def test_check_proxy_redacts_credentials(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://user:secret-password@proxy.corp:3128")
    result = _check_proxy(_local_env(80))
    assert "secret-password" not in result.detail
    assert "user" not in result.detail
    assert "HTTP_PROXY = http://****@proxy.corp:3128" in result.detail


def test_check_connectivity_ok_against_listening_port():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        result = _check_connectivity(_local_env(port))
    assert result.status is CheckStatus.OK
    assert f"127.0.0.1:{port} reachable" in result.detail


def test_check_connectivity_fails_on_closed_port():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
    result = _check_connectivity(_local_env(port))
    assert result.status is CheckStatus.FAIL
    assert "TCP connect failed" in result.detail
    assert "IP allowlist" in result.fix


def test_check_connectivity_fails_on_dns_error(monkeypatch):
    def raise_gaierror(*args, **kwargs):
        raise socket.gaierror("Name or service not known")

    monkeypatch.setattr(diagnostics.socket, "getaddrinfo", raise_gaierror)
    result = _check_connectivity(_local_env(80))
    assert result.status is CheckStatus.FAIL
    assert "DNS resolution failed" in result.detail
    assert "VPN" in result.fix


def test_check_connectivity_skips_proxied_hosts(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.corp:3128")
    result = _check_connectivity(_local_env(1))
    assert result.status is CheckStatus.OK
    assert "probe skipped (proxy in use)" in result.detail
    # The finding line must not claim unprobed hosts are reachable.
    finding = result.detail.splitlines()[0]
    assert "0 of 1 hosts reachable" in finding
    assert "1 not probed directly (proxy in use)" in finding


def test_check_connectivity_ok_finding_counts_all_probed_hosts():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        result = _check_connectivity(_local_env(port))
    assert result.detail.splitlines()[0] == "all 1 hosts reachable"


def test_check_tls_skips_plain_http():
    result = _check_tls(_local_env(80))
    assert result.status is CheckStatus.SKIP
    assert "no https endpoints" in result.detail


def test_ca_bundle_matches_requests_resolution_order(monkeypatch, tmp_path):
    requests_bundle = tmp_path / "requests.pem"
    curl_bundle = tmp_path / "curl.pem"
    requests_bundle.write_text("")
    curl_bundle.write_text("")
    for name in ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE"):
        monkeypatch.delenv(name, raising=False)

    default_bundle = diagnostics._ca_bundle()
    assert default_bundle.endswith(".pem")

    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "ignored.pem"))
    assert diagnostics._ca_bundle() == default_bundle

    monkeypatch.setenv("CURL_CA_BUNDLE", str(curl_bundle))
    assert diagnostics._ca_bundle() == str(curl_bundle)

    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(requests_bundle))
    assert diagnostics._ca_bundle() == str(requests_bundle)


def test_check_http_detects_html_fallthrough(monkeypatch):
    def fake_get(url, timeout):
        return _StubResponse(
            status_code=200,
            headers={"Content-Type": "text/html"},
            text="<!DOCTYPE html><html><body>app</body></html>",
        )

    monkeypatch.setattr(diagnostics, "_client_session", lambda: SimpleNamespace(get=fake_get))
    result = _check_http(_local_env(80), {})
    assert result.status is CheckStatus.FAIL
    assert "returned HTML" in result.detail
    assert "flow360-api" in result.fix


def test_check_http_ok_and_stores_response(monkeypatch):
    response = _StubResponse(status_code=200, headers={"Content-Type": "application/json"})
    monkeypatch.setattr(
        diagnostics, "_client_session", lambda: SimpleNamespace(get=lambda url, timeout: response)
    )
    state = {}
    result = _check_http(_local_env(80), state)
    assert result.status is CheckStatus.OK
    assert state["health_response"] is response


def test_check_http_warns_on_non_200(monkeypatch):
    monkeypatch.setattr(
        diagnostics,
        "_client_session",
        lambda: SimpleNamespace(get=lambda url, timeout: _StubResponse(status_code=404)),
    )
    result = _check_http(_local_env(80), {})
    assert result.status is CheckStatus.WARN
    assert "404" in result.detail


def test_check_http_fails_on_connection_error(monkeypatch):
    def fake_get(url, timeout):
        raise diagnostics.requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(diagnostics, "_client_session", lambda: SimpleNamespace(get=fake_get))
    result = _check_http(_local_env(80), {})
    assert result.status is CheckStatus.FAIL
    assert "failed: refused" in result.detail


def test_check_http_fails_on_ssl_error(monkeypatch):
    def fake_get(url, timeout):
        raise diagnostics.requests.exceptions.SSLError("bad handshake")

    monkeypatch.setattr(diagnostics, "_client_session", lambda: SimpleNamespace(get=fake_get))
    result = _check_http(_local_env(443, scheme="https"), {})
    assert result.status is CheckStatus.FAIL
    assert "[tls]" in result.fix


def test_check_clock_statuses():
    from datetime import datetime, timedelta, timezone
    from email.utils import format_datetime

    def result_for(offset_seconds):
        stamp = datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)
        response = _StubResponse(headers={"Date": format_datetime(stamp, usegmt=True)})
        return _check_clock({"health_response": response})

    assert result_for(0).status is CheckStatus.OK
    assert result_for(-300).status is CheckStatus.WARN
    assert result_for(1200).status is CheckStatus.FAIL

    assert _check_clock({}).status is CheckStatus.SKIP
    assert _check_clock({"health_response": _StubResponse()}).status is CheckStatus.SKIP


def test_transport_checks_skip_cascade_on_connectivity_failure(monkeypatch):
    def raise_gaierror(*args, **kwargs):
        raise socket.gaierror("Name or service not known")

    monkeypatch.setattr(diagnostics.socket, "getaddrinfo", raise_gaierror)
    runner = _CheckRunner()
    _run_transport_checks(runner, _local_env(80), "in-memory")

    statuses = {check.name: check.status for check in runner.report.checks}
    assert statuses["connectivity"] is CheckStatus.FAIL
    assert statuses["tls"] is CheckStatus.SKIP
    assert statuses["http"] is CheckStatus.SKIP
    assert statuses["clock"] is CheckStatus.SKIP
    assert not runner.report.ok


def test_check_api_key_warns_on_environment_variable_override(monkeypatch):
    monkeypatch.setenv("FLOW360_APIKEY", "abcdefghijklmnop")
    result = _check_api_key(_local_env(80))
    assert result.status is CheckStatus.WARN
    assert "abcd...mnop" in result.detail
    assert "not environment-aware" in result.detail
    assert "abcdefghijklmnop" not in result.detail
    assert "flow360 configure --env local --apikey <apikey>" in result.fix


def test_check_api_key_reports_config_section_for_custom_env(monkeypatch):
    monkeypatch.delenv("FLOW360_APIKEY", raising=False)
    monkeypatch.setattr("flow360.cloud.security.api_key", lambda: "abcdefghijklmnop")
    result = _check_api_key(_local_env(80))
    assert result.status is CheckStatus.OK
    assert "config.toml [default][local]" in result.detail


def test_check_api_key_missing(monkeypatch):
    monkeypatch.delenv("FLOW360_APIKEY", raising=False)
    monkeypatch.setattr("flow360.cloud.security.api_key", lambda: None)
    result = _check_api_key(_local_env(80))
    assert result.status is CheckStatus.FAIL
    assert "No API key configured" in result.fix
    assert "flow360 login" in result.fix


def test_check_webapi_auth_ok(monkeypatch):
    monkeypatch.setattr(
        "flow360.cloud.http_util.http.get", lambda url: {"email": "user@example.com"}
    )
    result = _check_webapi_auth(_local_env(80))
    assert result.status is CheckStatus.OK
    assert "authenticated as user@example.com" in result.detail


def test_check_webapi_auth_classifies_401(monkeypatch):
    def raise_401(url):
        raise Flow360AuthorisationError("Web url: Unauthorized")

    monkeypatch.setattr("flow360.cloud.http_util.http.get", raise_401)
    result = _check_webapi_auth(_local_env(80))
    assert result.status is CheckStatus.FAIL
    assert "401" in result.detail
    assert "environment-specific" in result.fix
    assert "[apikey] check above" in result.fix
    assert "flow360 configure --env local --apikey <apikey>" in result.fix


def test_runner_silences_client_exception_logs(caplog):
    import logging

    runner = _CheckRunner()

    def provoke_expected_failure():
        raise Flow360AuthorisationError("Web url: Unauthorized")

    with caplog.at_level(logging.ERROR, logger="flow360_schema"):
        result = runner.run("auth", provoke_expected_failure)

    assert result.status is CheckStatus.FAIL
    assert not caplog.records


def test_check_webapi_auth_classifies_403_ip_restriction(monkeypatch):
    def raise_403(url):
        raise Flow360WebError(
            "Web url: Unexpected response error: 403:"
            " Please make sure you are under the specific VPN."
        )

    monkeypatch.setattr("flow360.cloud.http_util.http.get", raise_403)
    result = _check_webapi_auth(_local_env(80))
    assert result.status is CheckStatus.FAIL
    assert "403" in result.detail
    assert "specific VPN" in result.detail
    assert "VPN" in result.fix


def test_check_webapi_auth_classifies_404(monkeypatch):
    def raise_404(url):
        raise Flow360WebNotFoundError("Web url: Not found error")

    monkeypatch.setattr("flow360.cloud.http_util.http.get", raise_404)
    result = _check_webapi_auth(_local_env(80))
    assert result.status is CheckStatus.FAIL
    assert "404" in result.detail
    assert "wrong service or path" in result.fix


def test_check_webapi_auth_fails_on_non_json_answer(monkeypatch):
    monkeypatch.setattr("flow360.cloud.http_util.http.get", lambda url: None)
    result = _check_webapi_auth(_local_env(80))
    assert result.status is CheckStatus.FAIL
    assert "web_api_endpoint" in result.fix


def test_check_portal_auth_ok(monkeypatch):
    monkeypatch.setattr(
        "flow360.cloud.http_util.http.portal_api_get",
        lambda path: {"email": "user@example.com"},
    )
    result = _check_portal_auth(_local_env(80))
    assert result.status is CheckStatus.OK
    assert "user@example.com" in result.detail


def test_check_portal_auth_detects_wrong_portal_url(monkeypatch):
    monkeypatch.setattr("flow360.cloud.http_util.http.portal_api_get", lambda path: None)
    result = _check_portal_auth(_local_env(80))
    assert result.status is CheckStatus.FAIL
    assert "flow360-portal-api" in result.fix


def test_check_portal_auth_classifies_html_error_body(monkeypatch):
    import json

    def raise_json_decode_error(path):
        # http_interceptor's 400/404 branches call resp.json() unguarded; an
        # HTML error page from a wrong portal URL escapes as JSONDecodeError.
        raise json.JSONDecodeError("Expecting value", "<html></html>", 0)

    monkeypatch.setattr("flow360.cloud.http_util.http.portal_api_get", raise_json_decode_error)
    result = _check_portal_auth(_local_env(80))
    assert result.status is CheckStatus.FAIL
    assert "answered without JSON data" in result.detail
    assert "flow360-portal-api" in result.fix


def test_check_webapi_auth_classifies_html_error_body(monkeypatch):
    import json

    def raise_json_decode_error(url):
        raise json.JSONDecodeError("Expecting value", "<html></html>", 0)

    monkeypatch.setattr("flow360.cloud.http_util.http.get", raise_json_decode_error)
    result = _check_webapi_auth(_local_env(80))
    assert result.status is CheckStatus.FAIL
    assert "answered without JSON data" in result.detail
    assert "web_api_endpoint" in result.fix


def test_check_project_list_stores_records(monkeypatch):
    records = SimpleNamespace(records=[])
    monkeypatch.setattr(
        "flow360.component.simulation.web.project_records.get_project_records",
        lambda: (records, 7),
    )
    state = {}
    result = _check_project_list(state)
    assert result.status is CheckStatus.OK
    assert "7 projects visible" in result.detail
    assert state["project_records"] is records


def test_check_storage_fails_on_unreachable_override(monkeypatch):
    def raise_connection_error(url, timeout):
        raise diagnostics.requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(
        diagnostics, "_client_session", lambda: SimpleNamespace(get=raise_connection_error)
    )
    env = EnvironmentConfig.from_on_premises_url(name="onprem", base_url="http://localhost")
    result = _check_storage(env, {})
    assert result.status is CheckStatus.FAIL
    assert "not reachable" in result.detail
    assert "{base_url}/s3" in result.fix


def test_check_storage_without_override_or_projects():
    env = _local_env(80)
    assert env.s3_endpoint_url is None
    result = _check_storage(env, {})
    assert result.status is CheckStatus.OK
    assert "server-advertised endpoint is used" in result.detail
    assert "no existing project to probe" in result.detail


def test_check_storage_fails_on_internal_advertised_endpoint(monkeypatch):
    monkeypatch.setattr(
        diagnostics, "_advertised_storage_endpoint", lambda records: "http://s3proxy:8080"
    )

    def raise_gaierror(*args, **kwargs):
        raise socket.gaierror("Name or service not known")

    monkeypatch.setattr(diagnostics.socket, "getaddrinfo", raise_gaierror)
    state = {"project_records": SimpleNamespace(records=[SimpleNamespace(project_id="p")])}
    result = _check_storage(_local_env(80), state)
    assert result.status is CheckStatus.FAIL
    assert "does NOT resolve" in result.detail
    assert "s3_endpoint_url" in result.fix


def test_check_storage_ok_when_override_covers_internal_endpoint(monkeypatch):
    monkeypatch.setattr(
        diagnostics, "_advertised_storage_endpoint", lambda records: "http://s3proxy:8080"
    )
    monkeypatch.setattr(
        diagnostics,
        "_client_session",
        lambda: SimpleNamespace(get=lambda url, timeout: _StubResponse(status_code=403)),
    )

    def raise_gaierror(*args, **kwargs):
        raise socket.gaierror("Name or service not known")

    monkeypatch.setattr(diagnostics.socket, "getaddrinfo", raise_gaierror)
    env = EnvironmentConfig.from_on_premises_url(name="onprem", base_url="http://localhost")
    state = {"project_records": SimpleNamespace(records=[SimpleNamespace(project_id="p")])}
    result = _check_storage(env, state)
    assert result.status is CheckStatus.OK
    assert "override is required and in place" in result.detail


def test_service_checks_skip_cascade_on_missing_api_key(monkeypatch):
    monkeypatch.delenv("FLOW360_APIKEY", raising=False)
    monkeypatch.setattr("flow360.cloud.security.api_key", lambda: None)
    runner = _CheckRunner()
    runner.report.checks.append(CheckResult(name="connectivity", status=CheckStatus.OK, detail=""))
    _run_service_checks(runner, _local_env(80), {})

    statuses = {check.name: check.status for check in runner.report.checks}
    assert statuses["apikey"] is CheckStatus.FAIL
    assert statuses["webapi-auth"] is CheckStatus.SKIP
    assert statuses["portal-auth"] is CheckStatus.SKIP
    assert statuses["projects"] is CheckStatus.SKIP
    assert statuses["storage"] is CheckStatus.SKIP


def test_render_report_layout():
    report = DiagnosticReport(
        checks=[
            CheckResult(
                name="install", status=CheckStatus.OK, detail="flow360 x.y\nhidden evidence"
            ),
            CheckResult(
                name="connectivity",
                status=CheckStatus.FAIL,
                detail="line one\nline two",
                fix="do the thing " * 30,
            ),
        ]
    )
    rendered = _render_report(report)
    lines = rendered.splitlines()
    indent = " " * 8

    assert lines[0] == "[ 1/2] install      OK"
    assert lines[1] == indent + "flow360 x.y"
    assert "hidden evidence" not in rendered

    assert lines[2] == "[ 2/2] connectivity FAIL"
    assert lines[3] == indent + "line one"
    assert lines[4] == indent + "line two"
    assert lines[5].startswith(indent + "FIX: do the thing")
    # Long fix text is wrapped by the renderer; every non-header line keeps
    # the same indent so it cannot be mistaken for a check header.
    assert all(line.startswith(indent) or line.startswith("[") for line in lines[:-1])
    assert sum(line.startswith("[") for line in lines) == 2
    assert lines[-1] == "Some checks FAILED, see the FIX lines above."
