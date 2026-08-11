.. _python_api_folder_operations:

.. currentmodule:: flow360

*******************
Folder Operations
*******************

This example demonstrates how to work with folders in the Flow360 cloud platform using the Python API.
It shows how to create a parent folder and subfolders, how to search for a folder by name within a
folder tree, and how to submit a project directly into a specific folder.

The workflow is as follows:

- Create a root (parent) folder in your account.
- Create a child folder inside this parent folder.
- Define a helper function that walks the folder tree and returns the first folder whose name
  matches a given target.
- Use this helper to locate a folder by name starting from the root folder.
- Submit a new project from a geometry file into the located folder.

.. literalinclude:: _snippets/folder_operations.py
   :language: python
