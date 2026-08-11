.. _python_api_generate_multiple_meshes:

.. currentmodule:: flow360

**************************
Generate Multiple Meshes
**************************

This example demonstrates the procedure for generating multiple surface meshes from a single geometry using the Flow360 Python API. It illustrates uploading geometry via the ``Project`` interface, defining initial meshing parameters within ``SimulationParams``, and subsequently modifying these parameters to create distinct meshes within the same project context.

.. literalinclude:: _snippets/generate_multiple_meshes.py
   :language: python

Notes
=====

- Meshing parameters, contained within the ``MeshingParams`` class nested in ``SimulationParams``, can be adjusted between calls to ``project.generate_surface_mesh`` to explore different mesh configurations.
