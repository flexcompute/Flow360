.. _python_api_cli:

***
CLI
***

Installing the ``flow360`` Python package also installs a ``flow360`` command line
interface (CLI). The CLI exposes the most common project, asset, and account
operations directly from a terminal, so simulations can be configured, launched,
and inspected from shell scripts and continuous-integration jobs without writing
Python.

The CLI shares the same credentials and configuration file as the Python API
(``~/.flow360/config.toml``), so a project created with the CLI is visible in the
Python API and the :ref:`WebUI <workflowsInterfaces>`, and vice versa.

Most commands print their result as JSON so that the output can be piped into
tools such as ``jq``. A few commands (for example :ref:`project list
<cli_project>`) also offer a human-readable text format.

Authentication and configuration
=================================

Before using any command that talks to the cloud, you need to authenticate. This
is covered in detail in the :ref:`installation and setup guide
<python_api_setup>`; the relevant commands are summarized below.

.. _cli_login:

``flow360 login``
-----------------

Open a browser login flow and store the resulting API key automatically. This is
the recommended way to authenticate.

.. code-block:: bash

   flow360 login

Options:

* ``--profile`` : profile to store the resulting key under.
* ``--env <name>`` : log in to a named environment.

.. _cli_logout:

``flow360 logout``
------------------

Remove a stored API key.

.. code-block:: bash

   flow360 logout

Options ``--profile`` and ``--env`` select which stored credential to remove.

.. _cli_configure:

``flow360 configure``
---------------------

Store an API key directly, without the browser login flow. This is useful in
headless environments (such as continuous-integration jobs) where no browser is
available, and for setting account preferences.

.. code-block:: bash

   flow360 configure --apikey <YOUR_API_KEY>

Options:

* ``--apikey`` : the API key to store. If omitted (and none is stored yet) the
  command prompts for it interactively.
* ``--profile`` : store the key under a named profile (for example ``default`` or
  a secondary account) instead of the default profile.
* ``--env <name>`` : store the key for a named environment only.
* ``--suppress-submit-warning`` : toggle (``true``/``false``) the warnings shown
  when submitting a new ``Case``, ``VolumeMesh``, and similar resources.
* ``--beta-features`` : toggle (``true``/``false``) beta-feature support.

Running ``flow360 configure`` with no arguments and an existing key prints the
current configuration.

Getting help
============

Every command and group provides built-in help generated from the CLI itself:

.. code-block:: bash

   flow360 --help              # list all commands and command groups
   flow360 project --help      # list the subcommands of a group
   flow360 project create --help   # show the options of a single command

The ``--help`` output is always authoritative for the exact options available in
your installed version. Use :ref:`flow360 version <cli_version>` to check which
client version is installed.

Global options
==============

The following options apply to the root ``flow360`` command and are placed
**before** the subcommand. They select the profile and environment used for the
duration of the command:

.. code-block:: bash

   flow360 --profile my-profile project list

* ``--profile <name>`` : API key profile to use for requests.
* ``--env <name>`` : use a named environment.

.. _cli_resource_ids:

Resource identifiers
====================

Commands that act on an existing resource take its cloud identifier. Each
identifier carries a stable type prefix:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Prefix
     - Resource type
   * - ``prj-``
     - Project
   * - ``geo-``
     - Geometry
   * - ``sm-``
     - Surface mesh
   * - ``vm-``
     - Volume mesh
   * - ``case-``
     - Case
   * - ``dft-``
     - Draft
   * - ``folder-``
     - Folder

Commands that expect a specific type validate the prefix and report an error if
the wrong kind of ID is passed.

Account commands
================

.. _cli_show_projects:

``flow360 show_projects``
-------------------------

Display the available projects, with an optional keyword filter.

.. code-block:: bash

   flow360 show_projects
   flow360 show_projects --keyword wing

Options:

* ``--keyword`` / ``-k`` : filter projects by keyword.
* ``--env`` : environment to query.

.. _cli_version:

``flow360 version``
-------------------

Display the installed ``flow360`` client version together with the latest stable
and beta versions available for each solver release.

.. code-block:: bash

   flow360 version

.. _cli_project:

Projects
========

The ``flow360 project`` group inspects and manages projects.

``flow360 project create``
--------------------------

Create a new project from one or more input files.

.. code-block:: bash

   flow360 project create geometry.csm --from geometry --name "Wing study"

Arguments and options:

* ``FILES`` : one or more input files. Creating from a surface mesh or volume mesh
  expects exactly one file.
* ``--from`` : input file type, one of ``geometry``, ``surface-mesh``, or
  ``volume-mesh`` (required).
* ``--name`` : project name.
* ``--solver-version`` : solver version to use.
* ``--unit`` : project length unit (default ``m``).
* ``--tag`` : project tag. Repeat the option to attach multiple tags.
* ``--folder-id`` : destination folder ID.
* ``--workflow`` : geometry workflow, ``standard`` or ``catalyst`` (default
  ``standard``). Only used with ``--from geometry``.
* ``--sync`` : wait for the project root to finish processing before returning.

``flow360 project list``
------------------------

List projects.

.. code-block:: bash

   flow360 project list --limit 50 --format text

Options:

* ``--search`` / ``--keyword`` / ``-k`` : search project names.
* ``--limit`` / ``-n`` : maximum number of projects to return (1 to 1000, default
  25).
* ``--folder-id`` : restrict to one or more folder IDs. Repeat for multiple
  folders.
* ``--exclude-subfolders`` : search only the specified folders, not their
  subfolders.
* ``--format`` : output format, ``json`` or ``text`` (default ``json``).

Other project commands
-----------------------

* ``flow360 project info PROJECT_ID`` : print project metadata.
* ``flow360 project rename PROJECT_ID --name <new name>`` : rename a project.
* ``flow360 project delete PROJECT_ID --yes`` : delete a project. The ``--yes``
  flag is required to confirm.
* ``flow360 project tree PROJECT_ID`` : print the project tree.
* ``flow360 project items PROJECT_ID`` : print a flat list of project items.
* ``flow360 project path PROJECT_ID --item-id <id> --item-type <type>`` : print
  the path of items leading to a target item.

.. _cli_assets:

Assets
======

Geometries, surface meshes, volume meshes, and cases share a common set of
inspection commands, exposed under the ``geometry``, ``surface-mesh``,
``volume-mesh``, and ``case`` groups respectively. For an overview of these asset
types and how they relate in the project tree, see
:ref:`Asset Types <project_management>`.

Each group provides:

* ``info <ID>`` : print resource metadata.
* ``state <ID>`` : print the resource lifecycle state.
* ``rename <ID> --name <new name>`` : rename the resource.
* ``delete <ID> --yes`` : delete the resource (``--yes`` confirms).
* ``summary <ID>`` : summarize the validated ``SimulationParams`` user intent.
* ``simulation-params get <ID>`` : print the raw ``SimulationParams`` JSON.

For example:

.. code-block:: bash

   flow360 geometry info geo-xxxxxxxx
   flow360 volume-mesh state vm-xxxxxxxx
   flow360 case simulation-params get case-xxxxxxxx

Case results
------------

Cases expose an additional ``results`` group for result artifacts:

* ``flow360 case results list CASE_ID`` : list the available result artifacts.
* ``flow360 case results get CASE_ID RESULT_REF`` : download a result artifact.
  ``RESULT_REF`` can be the full path or the file basename.

Options for ``case results get``:

* ``--output`` / ``-o`` : output file path. Defaults to the result basename.
* ``--overwrite`` : overwrite an existing local file.

.. code-block:: bash

   flow360 case results list case-xxxxxxxx
   flow360 case results get case-xxxxxxxx total_forces_v2.csv -o forces.csv

.. _cli_draft:

Drafts
======

A draft holds an editable ``SimulationParams`` configuration before it is run.
The ``flow360 draft`` group creates drafts, edits their parameters, and launches
them.

* ``flow360 draft list --project-id PROJECT_ID`` : list the drafts in a project.
* ``flow360 draft create SOURCE_ID [--name <name>]`` : create a draft from a
  project or asset (``prj-``, ``geo-``, ``sm-``, ``vm-``, or ``case-`` ref).
* ``flow360 draft info DRAFT_ID`` / ``state DRAFT_ID`` : inspect a draft.
* ``flow360 draft rename DRAFT_ID --name <name>`` : rename a draft.
* ``flow360 draft delete DRAFT_ID --yes`` : delete a draft.

Editing draft parameters:

* ``flow360 draft simulation-params get DRAFT_ID`` : print the raw
  ``SimulationParams`` JSON.
* ``flow360 draft simulation-params set DRAFT_ID FILE`` : replace the
  ``SimulationParams`` from a JSON file.
* ``flow360 draft simulation-params patch DRAFT_ID PATCH_FILE`` : apply a local
  JSON merge patch. Recommended only for small edits such as angle of attack or
  velocity; use Python and the Pydantic models for larger, validation-sensitive
  edits.

Running a draft:

.. code-block:: bash

   flow360 draft run prj-xxxxxxxx params.json --up-to case --wait

``flow360 draft run SOURCE_ID [SIMULATION_PARAMS_FILE]`` creates or reuses a
draft, optionally sets its ``SimulationParams``, and runs it. Options:

* ``--up-to`` : target asset to run up to, one of ``surface-mesh``,
  ``volume-mesh``, or ``case`` (default ``case``).
* ``--name`` : draft/run name when ``SOURCE_ID`` is not already a draft.
* ``--patch`` : apply a small local JSON merge patch before running.
* ``--wait`` : wait for the run result.
* ``--timeout`` : wait timeout in seconds (default 3600).
* ``--poll-interval`` : wait polling interval in seconds (default 15).

.. _cli_folder:

Folders
=======

The ``flow360 folder`` group organizes projects into folders.

* ``flow360 folder get FOLDER_ID`` : print folder metadata.
* ``flow360 folder tree [FOLDER_ID]`` : print the folder tree (defaults to the
  root folder).
* ``flow360 folder create --name <name>`` : create a folder. Options:
  ``--parent-folder-id`` (defaults to the root folder) and ``--tag`` (repeatable).
* ``flow360 folder rename FOLDER_ID --name <name>`` : rename a folder.
* ``flow360 folder move FOLDER_ID --parent-folder-id <id>`` : move a folder under
  a new parent.
* ``flow360 folder delete FOLDER_ID --yes`` : delete a folder (``--yes`` confirms).

.. _cli_utility:

Utility commands
================

``flow360 open``
----------------

Open a resource in the browser.

.. code-block:: bash

   flow360 open prj-xxxxxxxx

The environment is selected through the root options, for example
``flow360 --env <name> open REF_ID``.

``flow360 wait``
----------------

Wait for a resource to reach a terminal state.

.. code-block:: bash

   flow360 wait case-xxxxxxxx --timeout 7200

Options:

* ``--timeout`` : maximum wait time in seconds (default 3600).
* ``--poll-interval`` : polling interval in seconds (default 2).

The command exits with status ``124`` if the timeout is reached, and ``1`` if the
resource finishes in a non-success state, which makes it convenient for use in
scripts.

``flow360 logs``
----------------

Fetch logs for a completed or running resource. Logs are supported for cases
(``case-``), volume meshes (``vm-``), and surface meshes (``sm-``).

.. code-block:: bash

   flow360 logs case-xxxxxxxx --tail 500
   flow360 logs case-xxxxxxxx --all --save case.log

Options (use at most one of the first three; the default is ``--tail 200``):

* ``--tail N`` : show the last N lines.
* ``--head N`` : show the first N lines.
* ``--all`` : show the entire log.
* ``--save FILE`` : write the selected output to a file instead of standard
  output.

.. seealso::

   * :ref:`Installation and setup <python_api_setup>` for installing the package
     and configuring an API key.
   * :ref:`Workflows and Interfaces <workflowsInterfaces>` for an overview of the
     CLI alongside the WebUI and Python API.
   * :doc:`Cloud Assets API reference </python_api/API_reference/cloud_assets>`
     for the Python classes behind these resources.
