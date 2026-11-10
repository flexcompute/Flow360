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

- ``project.import_surface_mesh(filename, name=)`` uploads the surface and registers it with the project. Pass the returned surfaces to ``fl.create_draft()`` through ``imported_surfaces`` before referencing them as ``draft.imported_surfaces["name"]``.
- Define the integral variable as the **local** quantity, as ``MassFlux`` does below. The area weighting is applied for you.

.. seealso::

   :ref:`sampleSurfaces_userGuide` for the supported file formats, which outputs accept a sample surface, the field restrictions, and the treatment of nodes outside the fluid domain.

   A fuller example with boundary conditions and user-defined integral variables: `import_surface_field_and_integral.py <https://github.com/flexcompute/Flow360/blob/main/examples/post_processing/imported_surfaces/import_surface_field_and_integral.py>`_.
