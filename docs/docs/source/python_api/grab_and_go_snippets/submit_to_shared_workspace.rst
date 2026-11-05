.. _python_api_submit_to_shared_workspace:

.. currentmodule:: flow360

**************************************
Submit a Project to a Shared Workspace
**************************************

This example demonstrates how to create a project in your company's shared workspace from the Flow360 Python API.
It selects the shared workspace, submits a pre-existing volume mesh into it, and then shows the same submission into a folder that lives in the shared workspace.
Every member of the company sees the resulting projects in the WebUI, exactly as if they had been created there.

.. literalinclude:: _snippets/submit_to_shared_workspace.py
   :language: python

Notes
=====

- ``Workspace.get_shared()`` raises an error when the shared workspace is not enabled for your company. ``Workspace.get_private()`` returns your private workspace, which is also the default when ``folder_or_workspace`` is omitted.
- ``folder_or_workspace`` is accepted by ``Project.from_geometry()``, ``Project.from_surface_mesh()`` and ``Project.from_volume_mesh()``; the workspace of an existing project can only be changed in the WebUI, see :doc:`GUI Guide: Workspaces </gui_guide/01.introduction/08.workspaces>`.
