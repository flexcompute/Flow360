.. _python_api_on_premises:

.. currentmodule:: flow360

****************************
On-Premises Deployment Setup
****************************

This page covers connecting the Python API to an **on-premises (Nexus)
deployment** of Flow360 instead of the public cloud. If you use
`flow360.simulation.cloud <https://flow360.simulation.cloud>`_, you do not need
anything on this page; follow the standard
:ref:`installation and setup guide <python_api_setup>`.

An on-premises deployment serves the web UI, the APIs, and object storage from a
single origin. The Python API needs the individual service endpoints, and
``EnvironmentConfig.from_on_premises_url`` derives all of them from one URL —
any address of the deployment, such as the web UI URL from your browser (only
the scheme, host and port are used):

.. code-block:: python

   import flow360 as fl
   from flow360.environment import EnvironmentConfig

   # Store the API key for this environment (obtain it from the deployment's
   # web UI under Account -> Python Authentication).
   fl.configure(apikey="<YOUR_API_KEY>", environment="my_on_premises")

   on_premises = EnvironmentConfig.from_on_premises_url(
       name="my_on_premises",
       base_url="http://nexus.example.com",  # any URL of the deployment works,
   )                                         # e.g. the web UI URL from the browser
   on_premises.active()

``from_on_premises_url`` derives:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Field
     - Derived value
   * - ``web_api_endpoint``
     - ``{base_url}/flow360-api``
   * - ``web_url``
     - ``{base_url}/flow360``
   * - ``portal_web_api_endpoint``
     - ``{base_url}/flow360-portal-api``
   * - ``s3_endpoint_url``
     - ``{base_url}/s3``

Object storage endpoint
=======================

On-premises deployments advertise an object-storage endpoint that is internal to
the deployment network (for example ``http://s3proxy:8080``), which client
machines cannot resolve. The ``s3_endpoint_url`` setting overrides it. The
default, ``{base_url}/s3``, routes uploads and downloads through the same origin
as everything else and works for both Docker Compose and Kubernetes deployments.
Two alternatives:

.. code-block:: python

   # A dedicated storage port, if your deployment publishes one (deploy-specific):
   EnvironmentConfig.from_on_premises_url(
       name="my_on_premises",
       base_url="http://nexus.example.com",
       s3_endpoint_url="http://nexus.example.com:9000",
   )

   # Trust the endpoint advertised by the server (only when the client can
   # resolve deployment-internal hostnames):
   EnvironmentConfig.from_on_premises_url(
       name="my_on_premises",
       base_url="http://nexus.example.com",
       s3_endpoint_url=None,
   )

Verify, then save
=================

Run the built-in diagnostic before persisting the environment, so a broken
configuration is never stored and silently reloaded in later sessions:

.. code-block:: python

   report = fl.diagnose()

   if report.ok:
       on_premises.save_config()   # persist to ~/.flow360/config.toml

``fl.diagnose()`` verifies one layer at a time — installation, endpoint
configuration, proxy environment, DNS and TCP reachability, TLS certificates,
HTTP, clock skew, API key, authenticated web API and portal calls, project
listing, and object storage — and prints a targeted fix for every failing
check. A saved environment can be re-activated in any later session with:

.. code-block:: python

   fl.Env.load("my_on_premises").active()

The Python setup above is a one-time step. Once the environment is saved,
verify it from the terminal at any time:

.. code-block:: bash

   flow360 diagnose --env my_on_premises

The ``environment`` name used with ``fl.configure`` and the environment name
must match: stored API keys are looked up per environment name. Both
``flow360 configure`` and ``flow360 diagnose`` also accept
``--profile <profile>`` to store and use the key under a non-default account
profile.

See the :ref:`CLI reference <cli_diagnose>` for all ``flow360 diagnose``
options.

Before running production work
==============================

.. admonition:: Important
   :class: warning

   - **Match the client version to the deployment.** Install a Python API
     version compatible with the Flow360 solver packaged in your deployment;
     mismatches surface as validation or submission failures. ``flow360
     version`` prints the installed client and its solver version.
   - **Confirm the object storage endpoint with your administrator.** An
     incorrect storage endpoint prevents uploads and downloads even when
     project listing works, because file transfers do not use the web API.
     The ``storage`` diagnostic check reports which endpoint is in use and
     whether the one advertised by the server is reachable from your machine.
   - **Verify one small upload and one download** against the deployment
     before working with production data. File transfers can behave
     differently between Docker Compose and Kubernetes deployments.
   - **Keep the API key confidential.** It is stored locally in
     ``~/.flow360/config.toml``. Do not commit a setup script that contains a
     real key, and do not share that configuration file.

Deployments behind HTTPS with a private certificate authority
==============================================================

If your deployment uses TLS certificates issued by an internal CA, point the
client at a PEM bundle that includes that CA before connecting:

.. code-block:: bash

   export REQUESTS_CA_BUNDLE=/path/to/internal-ca-bundle.pem

Some deployments are served over plain HTTP precisely because a self-signed
certificate is not trusted by the client machine. Match the scheme in
``base_url`` to what the deployment actually serves: use ``https`` only when
the deployment presents a certificate this machine trusts, either from a
public CA or through the bundle above.

The ``tls`` diagnostic check reports the CA bundle in use and distinguishes
certificate-verification failures from other connection problems.
