"""
Stepwise connection diagnostics for Flow360 cloud and on-premises deployments.

Each check verifies one layer of the stack (install, configuration, DNS/TCP,
TLS, HTTP, clock, authentication, object storage) and reports a targeted fix
on failure, so connection problems can be located without reading tracebacks.
"""

from __future__ import annotations

import logging
import os
import shutil
import socket
import ssl
import sys
import textwrap
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from enum import Enum
from importlib.util import find_spec
from typing import Callable, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from pydantic import BaseModel

from .environment import EnvironmentConfig
from .version import __version__

_TIMEOUT_SECONDS = 10
_CLOCK_WARN_SECONDS = 60
_CLOCK_FAIL_SECONDS = 600


class CheckStatus(str, Enum):
    """Outcome of a single diagnostic check."""

    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


class CheckResult(BaseModel):
    """Result of a single diagnostic check."""

    name: str
    status: CheckStatus
    detail: str = ""
    fix: Optional[str] = None


class DiagnosticReport(BaseModel):
    """Ordered results of a diagnostic run."""

    checks: List[CheckResult] = []

    @property
    def ok(self) -> bool:
        """True when no check failed."""
        return all(check.status is not CheckStatus.FAIL for check in self.checks)


def redact(secret: str) -> str:
    """Redact a secret, keeping the first and last four characters."""
    if len(secret) <= 12:
        return "****"
    return f"{secret[:4]}...{secret[-4:]}"


@contextmanager
def _quiet_expected_errors():
    """Silence client exception logs while a check runs.

    Flow360Error logs at ERROR on construction, so the expected failures the
    checks provoke and classify would additionally splash raw error blobs on
    the console. The report is the diagnostic output; drop the duplicates.
    """
    schema_logger = logging.getLogger("flow360_schema")
    previous_level = schema_logger.level
    schema_logger.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        schema_logger.setLevel(previous_level)


class _CheckRunner:  # pylint: disable=too-few-public-methods
    """Runs checks in order, skipping those whose prerequisites did not pass."""

    def __init__(self):
        self.report = DiagnosticReport()

    def _status_of(self, name: str) -> Optional[CheckStatus]:
        for check in self.report.checks:
            if check.name == name:
                return check.status
        return None

    def run(
        self, name: str, func: Callable[[], CheckResult], needs: Tuple[str, ...] = ()
    ) -> CheckResult:
        """Run one check unless a prerequisite failed or was skipped."""
        blocked = [
            need for need in needs if self._status_of(need) in (CheckStatus.FAIL, CheckStatus.SKIP)
        ]
        if blocked:
            result = CheckResult(
                name=name, status=CheckStatus.SKIP, detail=f"blocked by [{blocked[0]}]"
            )
        else:
            try:
                with _quiet_expected_errors():
                    result = func()
            except Exception as error:  # pylint: disable=broad-except
                result = CheckResult(
                    name=name,
                    status=CheckStatus.FAIL,
                    detail=f"unexpected {type(error).__name__}: {error}",
                )
        self.report.checks.append(result)
        return result


def _endpoint_urls(env: EnvironmentConfig) -> dict:
    urls = {
        "web_api_endpoint": env.web_api_endpoint,
        "web_url": env.web_url,
        "portal_web_api_endpoint": env.portal_web_api_endpoint,
        "s3_endpoint_url": env.s3_endpoint_url,
    }
    return {label: url for label, url in urls.items() if url}


def _host_port(url: str) -> Tuple[str, int]:
    parsed = urlparse(url)
    return parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)


def _uses_proxy(url: str) -> bool:
    proxies = requests.utils.get_environ_proxies(url, no_proxy=None)
    return requests.utils.select_proxy(url, proxies) is not None


def _ca_bundle() -> str:
    # Match requests' own resolution order (Session.merge_environment_settings):
    # REQUESTS_CA_BUNDLE, then CURL_CA_BUNDLE, then the bundled certifi store.
    return (
        os.environ.get("REQUESTS_CA_BUNDLE")
        or os.environ.get("CURL_CA_BUNDLE")
        or requests.certs.where()
    )


def _client_session():
    """The SDK's own session, so probes verify exactly as the client does."""
    # pylint: disable=import-outside-toplevel
    from .cloud.http_util import http

    return http.session


def _tls_context():
    """The trust configuration the client itself uses, plus a line describing it."""
    bundle = _ca_bundle()
    if os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("CURL_CA_BUNDLE"):
        # an explicit bundle is authoritative for the client too, so do not add the OS store
        return (
            ssl.create_default_context(cafile=bundle),
            f"CA bundle: {bundle} (explicit; OS trust store not consulted)",
        )
    # pylint: disable=import-outside-toplevel
    from .cloud._tls import system_ssl_context_or_none

    context = system_ssl_context_or_none()
    if context is None:
        return (
            ssl.create_default_context(cafile=bundle),
            f"CA bundle: {bundle} (OS trust store disabled by FLOW360_DISABLE_SYSTEM_CERTS)",
        )
    return context, f"CA bundle: {bundle} + OS trust store"


def _sanitize_proxy_value(value: str) -> str:
    """Redact credentials in authenticated proxy URLs (http://user:pass@host)."""
    if "@" not in value:
        return value
    prefix, _, host = value.rpartition("@")
    scheme_end = prefix.find("//")
    scheme = prefix[: scheme_end + 2] if scheme_end != -1 else ""
    return f"{scheme}****@{host}"


def _check_install() -> CheckResult:
    python_version = ".".join(str(part) for part in sys.version_info[:3])
    missing = [package for package in ("requests", "boto3") if find_spec(package) is None]
    if missing:
        return CheckResult(
            name="install",
            status=CheckStatus.FAIL,
            detail=f"missing required packages: {', '.join(missing)}",
            fix="Reinstall the client: pip install --upgrade flow360",
        )
    return CheckResult(
        name="install",
        status=CheckStatus.OK,
        detail=f"flow360 {__version__}, Python {python_version}",
    )


def _check_environment(env: EnvironmentConfig, source: str) -> CheckResult:
    evidence = [f"'{env.name}' resolved from {source}"]
    evidence += [f"{label} = {url}" for label, url in _endpoint_urls(env).items()]
    derive_fix = (
        "Recreate the environment from the deployment's web UI URL with"
        " EnvironmentConfig.from_on_premises_url() and save it once diagnose"
        " passes (see the on-premises setup guide)."
    )

    portal = env.portal_web_api_endpoint
    if portal is None:
        return CheckResult(
            name="environment",
            status=CheckStatus.WARN,
            detail="\n".join(
                ["portal_web_api_endpoint is not set: account and version checks will fail"]
                + evidence
            ),
            fix=derive_fix,
        )

    portal_parsed = urlparse(portal)
    web_parsed = urlparse(env.web_url)
    same_origin = (portal_parsed.hostname, portal_parsed.port) == (
        web_parsed.hostname,
        web_parsed.port,
    )
    if same_origin and portal_parsed.path.strip("/") == "":
        return CheckResult(
            name="environment",
            status=CheckStatus.WARN,
            detail="\n".join(
                [
                    "portal_web_api_endpoint looks like the web UI origin, not the portal"
                    " API (no /flow360-portal-api path)"
                ]
                + evidence
            ),
            fix=derive_fix,
        )
    return CheckResult(name="environment", status=CheckStatus.OK, detail="\n".join(evidence))


def _check_proxy(env: EnvironmentConfig) -> CheckResult:
    proxy_names = (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    )
    proxy_vars = {name: os.environ[name] for name in proxy_names if name in os.environ}
    if not proxy_vars:
        return CheckResult(
            name="proxy", status=CheckStatus.OK, detail="no proxy environment variables set"
        )

    evidence = "\n".join(
        f"{name} = {_sanitize_proxy_value(value)}" for name, value in proxy_vars.items()
    )
    proxied = [label for label, url in _endpoint_urls(env).items() if _uses_proxy(url)]
    if proxied:
        return CheckResult(
            name="proxy",
            status=CheckStatus.WARN,
            detail=f"requests to {', '.join(proxied)} will go through the proxy\n{evidence}",
            fix="If the deployment should be reached directly, add its host to NO_PROXY.",
        )
    return CheckResult(
        name="proxy",
        status=CheckStatus.OK,
        detail=f"proxy variables set but all configured endpoints bypass them\n{evidence}",
    )


def _check_connectivity(env: EnvironmentConfig) -> CheckResult:
    targets = {}
    for label, url in _endpoint_urls(env).items():
        targets.setdefault(_host_port(url), []).append((label, url))

    lines, failures, dns_failure, skipped = [], [], False, 0
    for (host, port), labelled_urls in sorted(targets.items()):
        if any(_uses_proxy(url) for _, url in labelled_urls):
            skipped += 1
            lines.append(f"{host}:{port} probe skipped (proxy in use)")
            continue
        try:
            socket.getaddrinfo(host, port)
        except OSError as error:
            dns_failure = True
            failures.append(host)
            lines.append(f"{host}:{port} DNS resolution failed ({error})")
            continue
        try:
            with socket.create_connection((host, port), timeout=_TIMEOUT_SECONDS):
                lines.append(f"{host}:{port} reachable")
        except OSError as error:
            failures.append(host)
            lines.append(f"{host}:{port} TCP connect failed ({error})")

    if failures:
        fix = (
            "Hostname does not resolve: check the URL, VPN, or /etc/hosts."
            if dns_failure
            else "Host resolved but the connection was not accepted: check that the"
            " deployment is running and that no firewall or network-level IP allowlist"
            " (security group, WAF, VPN requirement) blocks this machine."
        )
        return CheckResult(
            name="connectivity",
            status=CheckStatus.FAIL,
            detail="\n".join([f"{len(failures)} of {len(targets)} hosts unreachable"] + lines),
            fix=fix,
        )
    reachable = len(targets) - skipped
    if skipped:
        finding = (
            f"{reachable} of {len(targets)} hosts reachable,"
            f" {skipped} not probed directly (proxy in use);"
            " the [http] check below verifies the proxied path"
        )
    else:
        finding = f"all {len(targets)} hosts reachable"
    return CheckResult(
        name="connectivity",
        status=CheckStatus.OK,
        detail="\n".join([finding] + lines),
    )


def _check_tls(env: EnvironmentConfig) -> CheckResult:
    targets = {}
    for _, url in _endpoint_urls(env).items():
        if urlparse(url).scheme == "https" and not _uses_proxy(url):
            targets[_host_port(url)] = url
    if not targets:
        return CheckResult(
            name="tls", status=CheckStatus.SKIP, detail="no https endpoints to verify"
        )

    context, trust_detail = _tls_context()
    lines = []
    for host, port in sorted(targets):
        try:
            with socket.create_connection((host, port), timeout=_TIMEOUT_SECONDS) as sock:
                with context.wrap_socket(sock, server_hostname=host) as tls_sock:
                    cert = tls_sock.getpeercert()
        except ssl.SSLCertVerificationError as error:
            return CheckResult(
                name="tls",
                status=CheckStatus.FAIL,
                detail="\n".join(
                    [
                        f"{host}:{port} certificate verification failed:"
                        f" {error.verify_message or error}",
                        trust_detail,
                    ]
                    + lines
                ),
                fix="For deployments using a private CA or self-signed certificate, install"
                " the CA in the operating system trust store, or point REQUESTS_CA_BUNDLE at a"
                " PEM bundle that includes it. Re-run with FLOW360_DISABLE_SYSTEM_CERTS=1 to"
                " check whether the OS trust store is what is carrying this endpoint.",
            )
        except (OSError, ssl.SSLError) as error:
            return CheckResult(
                name="tls",
                status=CheckStatus.FAIL,
                detail="\n".join(
                    [f"{host}:{port} TLS handshake failed: {error}", trust_detail] + lines
                ),
                fix="The server did not complete a TLS handshake: confirm the endpoint"
                " really serves https on this port.",
            )
        subject = dict(item for pair in cert.get("subject", ()) for item in pair)
        issuer = dict(item for pair in cert.get("issuer", ()) for item in pair)
        lines.append(
            f"{host}:{port} certificate ok"
            f" (subject CN={subject.get('commonName', '?')},"
            f" issuer={issuer.get('organizationName', issuer.get('commonName', '?'))},"
            f" expires {cert.get('notAfter', '?')})"
        )
    return CheckResult(
        name="tls",
        status=CheckStatus.OK,
        detail="\n".join([f"{len(targets)} certificates verified ({trust_detail})"] + lines),
    )


def _check_http(env: EnvironmentConfig, state: dict) -> CheckResult:
    url = env.web_api_endpoint.rstrip("/") + "/health"
    try:
        response = _client_session().get(url, timeout=_TIMEOUT_SECONDS)
    except requests.exceptions.SSLError as error:
        return CheckResult(
            name="http",
            status=CheckStatus.FAIL,
            detail=f"GET {url} TLS error: {error}",
            fix="See the [tls] check above; the client session uses the same trust configuration.",
        )
    except requests.exceptions.RequestException as error:
        return CheckResult(
            name="http",
            status=CheckStatus.FAIL,
            detail=f"GET {url} failed: {error}",
            fix="The endpoint is not answering HTTP: check the URL scheme, port and"
            " any reverse proxy in front of the deployment.",
        )

    state["health_response"] = response
    body_start = response.text[:200].lstrip().lower()
    if "text/html" in response.headers.get("Content-Type", "") or body_start.startswith(
        ("<!doctype", "<html")
    ):
        return CheckResult(
            name="http",
            status=CheckStatus.FAIL,
            detail=f"GET {url} returned HTML instead of an API response",
            fix="The URL reaches a web page, not the API: web_api_endpoint is wrong."
            " On-premises deployments serve the API at {base_url}/flow360-api;"
            " recreate the environment with EnvironmentConfig.from_on_premises_url().",
        )
    if response.status_code == 200:
        return CheckResult(name="http", status=CheckStatus.OK, detail=f"GET {url} -> 200")
    return CheckResult(
        name="http",
        status=CheckStatus.WARN,
        detail=f"GET {url} -> {response.status_code}",
        fix="The API answered but not with 200 on /health; authenticated checks below"
        " will show whether this matters.",
    )


def _check_clock(state: dict) -> CheckResult:
    response = state.get("health_response")
    server_date = response.headers.get("Date") if response is not None else None
    if server_date is None:
        return CheckResult(
            name="clock", status=CheckStatus.SKIP, detail="server sent no Date header"
        )
    skew = abs((parsedate_to_datetime(server_date) - datetime.now(timezone.utc)).total_seconds())
    detail = f"clock skew vs server: {skew:.0f} s"
    if skew > _CLOCK_FAIL_SECONDS:
        return CheckResult(
            name="clock",
            status=CheckStatus.FAIL,
            detail=detail,
            fix="Synchronize the local clock (chrony/ntp): storage uploads sign requests"
            " with the local time and the server rejects large skews.",
        )
    if skew > _CLOCK_WARN_SECONDS:
        return CheckResult(name="clock", status=CheckStatus.WARN, detail=detail)
    return CheckResult(name="clock", status=CheckStatus.OK, detail=detail)


def _run_transport_checks(runner: _CheckRunner, env: EnvironmentConfig, source: str) -> dict:
    """Run checks 1-7 (install through clock); returns state shared with later checks."""
    state = {}
    runner.run("install", _check_install)
    runner.run("environment", lambda: _check_environment(env, source))
    runner.run("proxy", lambda: _check_proxy(env))
    runner.run("connectivity", lambda: _check_connectivity(env))
    runner.run("tls", lambda: _check_tls(env), needs=("connectivity",))
    runner.run("http", lambda: _check_http(env, state), needs=("connectivity",))
    runner.run("clock", lambda: _check_clock(state), needs=("http",))
    return state


def _classify_web_error(
    name: str, url: str, error: Exception, env: EnvironmentConfig
) -> CheckResult:
    """Map client web exceptions onto diagnostic results with targeted fixes."""
    # pylint: disable=import-outside-toplevel
    from .cli.auth_guidance import build_configure_command
    from .exceptions import Flow360AuthorisationError, Flow360WebNotFoundError
    from .user_config import UserConfig

    if isinstance(error, Flow360AuthorisationError):
        return CheckResult(
            name=name,
            status=CheckStatus.FAIL,
            detail=f"GET {url} -> 401 unauthorized",
            fix="The server rejected the API key; the [apikey] check above shows which"
            " key was used and where it came from. API keys are environment-specific:"
            f" copy the key from {env.web_url} (account section) and store it for this"
            f" environment with: {build_configure_command(env.name, UserConfig.profile)}",
        )
    # http_interceptor has no dedicated 403 branch; the status code only
    # survives in the generic message ("Unexpected response error: 403: ...").
    if ": 403" in str(error):
        return CheckResult(
            name=name,
            status=CheckStatus.FAIL,
            detail=f"GET {url} -> 403 forbidden: {error}",
            fix="The server refused access for this client's IP address. If your"
            " organization's account is IP-restricted, connect through your"
            " organization's VPN and retry.",
        )
    if isinstance(error, Flow360WebNotFoundError):
        return CheckResult(
            name=name,
            status=CheckStatus.FAIL,
            detail=f"GET {url} -> 404 not found",
            fix="The API host answered but this route does not exist there: the endpoint"
            " URL points at the wrong service or path.",
        )
    return CheckResult(name=name, status=CheckStatus.FAIL, detail=f"GET {url} failed: {error}")


def _check_api_key(env: EnvironmentConfig) -> CheckResult:
    # pylint: disable=import-outside-toplevel
    from .cli.auth_guidance import (
        build_configure_command,
        build_missing_api_key_message,
    )
    from .cloud.security import api_key
    from .environment import prod
    from .user_config import UserConfig

    key = api_key()
    profile = UserConfig.profile
    if not key:
        return CheckResult(
            name="apikey",
            status=CheckStatus.FAIL,
            detail=f"no API key configured for env '{env.name}', profile '{profile}'",
            fix=build_missing_api_key_message(env.name, profile),
        )
    if os.environ.get("FLOW360_APIKEY"):
        return CheckResult(
            name="apikey",
            status=CheckStatus.WARN,
            detail=f"{redact(key)} comes from FLOW360_APIKEY, which is not"
            " environment-aware and overrides the stored key for every environment",
            fix="Unset FLOW360_APIKEY and store the key per environment with:"
            f" {build_configure_command(env.name, profile)}",
        )
    if env.name == prod.name:
        source = f"config.toml [{profile}]"
    else:
        source = f"config.toml [{profile}][{env.name}]"
    return CheckResult(name="apikey", status=CheckStatus.OK, detail=f"{redact(key)} from {source}")


def _check_webapi_auth(env: EnvironmentConfig) -> CheckResult:
    # pylint: disable=import-outside-toplevel
    from .cloud.http_util import http
    from .exceptions import (
        Flow360AuthorisationError,
        Flow360WebError,
        Flow360WebNotFoundError,
    )

    url = env.get_real_url("v2/flow360/user")
    try:
        data = http.get(url)
    except (Flow360AuthorisationError, Flow360WebError, Flow360WebNotFoundError) as error:
        return _classify_web_error("webapi-auth", url, error, env)
    except ValueError:
        # The client crashes parsing a non-JSON error body (HTML error page):
        # the URL does not reach the API.
        data = None
    if not isinstance(data, dict):
        return CheckResult(
            name="webapi-auth",
            status=CheckStatus.FAIL,
            detail=f"GET {url} answered without JSON data",
            fix="The URL reaches something other than the Flow360 API:" " check web_api_endpoint.",
        )
    return CheckResult(
        name="webapi-auth",
        status=CheckStatus.OK,
        detail=f"authenticated as {data.get('email', '?')}",
    )


def _check_portal_auth(env: EnvironmentConfig) -> CheckResult:
    # pylint: disable=import-outside-toplevel
    from .cloud.http_util import http
    from .exceptions import (
        Flow360AuthorisationError,
        Flow360WebError,
        Flow360WebNotFoundError,
    )

    url = env.get_portal_real_url("auth/credential")
    try:
        data = http.portal_api_get("auth/credential")
    except (Flow360AuthorisationError, Flow360WebError, Flow360WebNotFoundError) as error:
        return _classify_web_error("portal-auth", url, error, env)
    except ValueError:
        # The client crashes parsing a non-JSON error body (HTML error page):
        # the URL does not reach the portal API.
        data = None
    if not isinstance(data, dict):
        return CheckResult(
            name="portal-auth",
            status=CheckStatus.FAIL,
            detail=f"GET {url} answered without JSON data (likely the web UI page)",
            fix="portal_web_api_endpoint does not point at the portal API. On-premises"
            " deployments serve it at {base_url}/flow360-portal-api;"
            " recreate the environment with EnvironmentConfig.from_on_premises_url().",
        )
    return CheckResult(
        name="portal-auth",
        status=CheckStatus.OK,
        detail=f"auth/credential answered for {data.get('email', '?')}",
    )


def _check_project_list(state: dict) -> CheckResult:
    # pylint: disable=import-outside-toplevel
    from .component.simulation.web.project_records import get_project_records

    records, total = get_project_records()
    state["project_records"] = records
    return CheckResult(name="projects", status=CheckStatus.OK, detail=f"{total} projects visible")


def _advertised_storage_endpoint(records) -> Optional[str]:
    """Fetch one read-only STS grant and return the storage endpoint it advertises."""
    # pylint: disable=import-outside-toplevel
    from .cloud.rest_api import RestApi
    from .cloud.s3_utils import S3TransferType
    from .environment import current_environment

    project_id = records.records[0].project_id
    metadata = RestApi("v2/projects", id=project_id, environment_provider=current_environment).get()
    transfer_type = {
        "Geometry": S3TransferType.GEOMETRY,
        "SurfaceMesh": S3TransferType.SURFACE_MESH,
        "VolumeMesh": S3TransferType.VOLUME_MESH,
    }[metadata["rootItemType"]]
    # pylint: disable=protected-access
    grant_path = transfer_type._get_grant_url(metadata["rootItemId"], "diagnostics-probe")
    grant = RestApi(grant_path, environment_provider=current_environment).get()
    return grant["userCredentials"].get("endpoint")


def _check_storage(env: EnvironmentConfig, state: dict) -> CheckResult:
    lines = []
    if env.s3_endpoint_url is not None:
        try:
            response = _client_session().get(env.s3_endpoint_url, timeout=_TIMEOUT_SECONDS)
            lines.append(f"override {env.s3_endpoint_url} reachable (HTTP {response.status_code})")
        except requests.exceptions.RequestException as error:
            return CheckResult(
                name="storage",
                status=CheckStatus.FAIL,
                detail=f"override {env.s3_endpoint_url} not reachable: {error}",
                fix="s3_endpoint_url is set but does not answer. For on-premises"
                " deployments use the {base_url}/s3 route or the storage host port"
                " configured at deploy time (the s3_endpoint_url argument of"
                " EnvironmentConfig.from_on_premises_url).",
            )
    else:
        lines.append("no s3_endpoint_url override; the server-advertised endpoint is used")

    records = state.get("project_records")
    if records is None or not records.records:
        lines.append("no existing project to probe a storage grant with")
        return CheckResult(name="storage", status=CheckStatus.OK, detail="\n".join(lines))

    advertised = _advertised_storage_endpoint(records)
    if advertised is None:
        lines.append("server advertises the AWS S3 default endpoint")
        return CheckResult(name="storage", status=CheckStatus.OK, detail="\n".join(lines))

    lines.append(f"server advertises {advertised}")
    host, port = _host_port(advertised)
    try:
        socket.getaddrinfo(host, port)
        lines.append(f"{host} resolves from this machine")
    except OSError:
        if env.s3_endpoint_url is None:
            return CheckResult(
                name="storage",
                status=CheckStatus.FAIL,
                detail="\n".join(
                    [
                        f"server-advertised storage endpoint host {host} does NOT resolve"
                        " from this machine and no s3_endpoint_url override is set"
                    ]
                    + lines
                ),
                fix="The server advertises a storage endpoint that is internal to the"
                " deployment. Set s3_endpoint_url to the deployment's /s3 route"
                " (EnvironmentConfig.from_on_premises_url defaults it to {origin}/s3).",
            )
        lines.append(f"{host} does not resolve here; the override is required and in place")
    return CheckResult(name="storage", status=CheckStatus.OK, detail="\n".join(lines))


def _run_service_checks(runner: _CheckRunner, env: EnvironmentConfig, state: dict) -> None:
    """Run checks 8-12 (API key through storage) via the real client code paths."""
    runner.run("apikey", lambda: _check_api_key(env))
    runner.run("webapi-auth", lambda: _check_webapi_auth(env), needs=("apikey", "connectivity"))
    runner.run("portal-auth", lambda: _check_portal_auth(env), needs=("apikey", "connectivity"))
    runner.run("projects", lambda: _check_project_list(state), needs=("webapi-auth",))
    runner.run("storage", lambda: _check_storage(env, state), needs=("webapi-auth",))


_REPORT_INDENT = " " * 8


def _render_report(report: DiagnosticReport) -> str:
    # WARN/FAIL details go on their own lines under the check header, wrapped
    # to the terminal width with one uniform indent, so continuation lines
    # never fall back to column 0 and get mistaken for check headers.
    width = max(shutil.get_terminal_size((100, 24)).columns, 60)

    def wrapped(text: str) -> List[str]:
        return textwrap.wrap(
            text, width=width, initial_indent=_REPORT_INDENT, subsequent_indent=_REPORT_INDENT
        ) or [""]

    total = len(report.checks)
    lines = []
    for index, check in enumerate(report.checks, start=1):
        lines.append(f"[{index:>2}/{total}] {check.name:<12} {check.status.value}")
        detail_lines = check.detail.splitlines() or [""]
        if check.status in (CheckStatus.OK, CheckStatus.SKIP):
            # Healthy and skipped checks stay quiet: the finding line only.
            # Evidence lines remain in the report object and the JSON output.
            detail_lines = detail_lines[:1]
        for detail_line in detail_lines:
            lines.extend(wrapped(detail_line))
        if check.fix:
            lines.extend(wrapped(f"FIX: {check.fix}"))
    lines.append(
        "All checks passed." if report.ok else "Some checks FAILED, see the FIX lines above."
    )
    return "\n".join(lines)


def diagnose(
    env: Optional[EnvironmentConfig] = None,
    *,
    source: Optional[str] = None,
    print_report: bool = True,
) -> DiagnosticReport:
    """Run stepwise connection diagnostics against a Flow360 deployment.

    Verifies one layer at a time (install, configuration, proxy, DNS/TCP, TLS,
    HTTP, clock, API key, authenticated web API and portal calls, project
    listing, object storage) and prints a report with a targeted fix for every
    failing check.

    Parameters
    ----------
    env : EnvironmentConfig, optional
        Environment to diagnose. Defaults to the currently active environment;
        when given, it is activated for the duration of the run and the
        previous environment is restored afterwards.
    source : str, optional
        Label describing where the environment came from, used in the report.
    print_report : bool, optional
        Print the rendered report to stdout (default True).

    Returns
    -------
    DiagnosticReport
        All check results; ``report.ok`` is True when nothing failed.
    """
    # pylint: disable=import-outside-toplevel
    from .environment import Env

    previous = None
    if env is None:
        env = Env.current
        source = source or "the active environment"
    else:
        previous = Env.current
        env.active()
        source = source or "function argument"

    try:
        runner = _CheckRunner()
        state = _run_transport_checks(runner, env, source)
        _run_service_checks(runner, env, state)
    finally:
        if previous is not None:
            previous.active()

    if print_report:
        print(_render_report(runner.report))
    return runner.report
