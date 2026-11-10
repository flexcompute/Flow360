.. _python_api_download_results:

.. currentmodule:: flow360

**********************************
Download Results
**********************************

This example demonstrates how to retrieve simulation results from the Flow360 platform using a specified Case ID.
It facilitates the download of surface and volume data to a local directory and subsequently extracts the compressed files.

.. literalinclude:: _snippets/download_results.py
   :language: python

Notes
=====

- Boolean flags (``download_surfaces``, ``download_volumes``) control which result types are retrieved.
- Downloaded ``.tar.gz`` archives are automatically extracted into subdirectories, and the original archives are removed.
