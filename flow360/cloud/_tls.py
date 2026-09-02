"""OS trust store support, scoped to this client's own transports.

The process-global ssl module is left untouched, so a script's own requests
calls keep whatever trust they had.
"""

import os
import ssl

import requests
import truststore
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context

# urllib3 applies these to the context it builds; a context we build ourselves has to
# carry them across or it silently drifts from stock requests. hostname_checks_common_name
# is the load-bearing one: leaving it True accepts a CN-only certificate with no SAN.
_POLICY_ATTRS = (
    "options",
    "verify_flags",
    "minimum_version",
    "maximum_version",
    "post_handshake_auth",
    "hostname_checks_common_name",
)


def system_certs_enabled():
    """Whether OS trust store verification is on; opt out before importing flow360."""
    return os.environ.get("FLOW360_DISABLE_SYSTEM_CERTS", "").lower() not in ("1", "true", "yes")


def _os_trust_context(reference):
    """A context verifying against the OS trust store, carrying reference's TLS policy.

    truststore reaches every platform store: SecTrust on macOS, ROOT/CA on Windows, and
    OpenSSL's default paths (with a distro CA-file fallback) on Linux. Its context cannot
    be built by urllib3's factory, so the policy that factory applied to reference is
    copied over. certifi stays additive underneath on every platform.
    """
    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    for name in _POLICY_ATTRS:
        setattr(context, name, getattr(reference, name))
    context.load_verify_locations(cafile=requests.certs.where())
    return context


def system_ssl_context():
    """A fresh context trusting the OS store additively over certifi."""
    return _os_trust_context(create_urllib3_context())


def system_ssl_context_or_none():
    """The shared trust context, or None when the OS store is opted out.

    None leaves the caller on its own stock default, which is what opting out means.
    Callers outside this module should use this rather than system_ssl_context(), so
    FLOW360_DISABLE_SYSTEM_CERTS stays honoured everywhere.
    """
    return system_ssl_context() if system_certs_enabled() else None


class SystemTrustAdapter(HTTPAdapter):
    """requests adapter verifying certificates against the OS trust store."""

    def __init__(self, *args, **kwargs):
        """Track the per-client-cert contexts and the lazily built verify=False adapter."""
        self._context_cache = {}
        self._insecure_adapter = None
        super().__init__(*args, **kwargs)

    def init_poolmanager(self, *args, **kwargs):
        """Give the pool the shared OS-trust context unless one was passed in."""
        kwargs.setdefault("ssl_context", system_ssl_context())
        return super().init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, proxy, **proxy_kwargs):
        """Trust the OS store for the proxy handshake too.

        The proxy and endpoint get separate contexts: urllib3 loads a connection's
        ca_certs into whichever context it is handed, so one shared object would let
        the two handshakes contaminate each other's trust configuration.
        """
        if proxy in self.proxy_manager:
            return self.proxy_manager[proxy]
        if proxy.lower().startswith("https"):
            proxy_kwargs.setdefault("proxy_ssl_context", system_ssl_context())
        proxy_kwargs.setdefault("ssl_context", system_ssl_context())
        return super().proxy_manager_for(proxy, **proxy_kwargs)

    def get_connection_with_tls_context(self, request, verify, proxies=None, cert=None):
        """Route verify=False to a stock adapter; the rest keep the OS-trust context."""
        # CERT_NONE cannot be set on a check_hostname context, so delegate to a stock adapter
        if verify is False:
            if self._insecure_adapter is None:
                self._insecure_adapter = HTTPAdapter(
                    pool_connections=self._pool_connections,
                    pool_maxsize=self._pool_maxsize,
                    pool_block=self._pool_block,
                )
            return self._insecure_adapter.get_connection_with_tls_context(
                request, verify, proxies, cert
            )
        return super().get_connection_with_tls_context(request, verify, proxies, cert)

    def close(self):
        """Also drop the per-client-cert contexts and the verify=False adapter."""
        super().close()
        self._context_cache.clear()
        if self._insecure_adapter is not None:
            self._insecure_adapter.close()

    def build_connection_pool_key_attributes(self, request, verify, cert=None):
        """Keep an explicit bundle authoritative and client certs off the shared context."""
        host_params, pool_kwargs = super().build_connection_pool_key_attributes(
            request, verify, cert
        )
        if "ca_certs" in pool_kwargs or "ca_cert_dir" in pool_kwargs:
            # an explicit bundle stays authoritative: None unsets the shared context so
            # urllib3 builds a stock one from the bundle alone
            pool_kwargs["ssl_context"] = None
        elif "cert_file" in pool_kwargs:
            # own context per client-cert config, else it leaks into the shared one; cached to keep pooling
            config = tuple(
                (key, pool_kwargs[key]) for key in ("cert_file", "key_file") if key in pool_kwargs
            )
            if config not in self._context_cache:
                self._context_cache[config] = system_ssl_context()
            pool_kwargs["ssl_context"] = self._context_cache[config]
        return host_params, pool_kwargs


def make_session():
    """A requests.Session verifying against the OS trust store plus certifi."""
    session = requests.Session()
    if system_certs_enabled():
        session.mount("https://", SystemTrustAdapter())
    return session


def inject_system_trust(client):
    """Retarget a boto3 client's existing http session, preserving its config."""
    if not system_certs_enabled():
        return client
    # pylint: disable=protected-access
    session = client._endpoint.http_session
    if session._verify is not True:
        # verify=False or an explicit bundle (e.g. AWS_CA_BUNDLE) stays authoritative
        return client
    original_ssl_context = session._get_ssl_context

    def patched_ssl_context():
        # keep botocore's own TLS policy, only add the OS store on top
        return _os_trust_context(original_ssl_context())

    session._get_ssl_context = patched_ssl_context
    original_proxy_ssl_context = session._setup_proxy_ssl_context
    session._setup_proxy_ssl_context = lambda proxy_url: (
        original_proxy_ssl_context(proxy_url) or patched_ssl_context()
    )
    # only the eager proxy manager needs this; lazy ones read _get_ssl_context
    session._manager.connection_pool_kw["ssl_context"] = patched_ssl_context()
    return client
