.. _python_api_import_sample_surface:

.. currentmodule:: flow360

*********************
Import Sample Surface
*********************

This snippet demonstrates how to import external surface mesh files into a Flow360 project
and use them as output surfaces to extract flow field data and compute surface integrals.

.. literalinclude:: _snippets/import_sample_surface.py
   :language: python

Notes
=====

- ``project.import_surface_mesh(filename, name=)`` uploads the surface file to the cloud and registers it with the project. The ``name`` argument is the key used to reference the surface inside the draft context.
- Imported surfaces must be passed to ``fl.create_draft()`` via the ``imported_surfaces`` argument before they can be referenced.
- Inside the draft context, use ``draft.imported_surfaces["name"]`` to retrieve a reference to an imported surface for use in any output type.
- Supported surface file formats are STL, CGNS, and UGRID.
- For a complete example that includes boundary conditions and user-defined surface integral variables, see the `Flow360 GitHub example <https://github.com/flexcompute/Flow360/blob/main/examples/post_processing/imported_surfaces/import_surface_field_and_integral.py>`_.
