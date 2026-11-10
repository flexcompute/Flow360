.. _python_api_compute_projected_area:

.. currentmodule:: flow360

**********************************
Compute a projected reference area
**********************************

This example previews the projected silhouette area of selected Geometry
surfaces and records an automatic projected-area recipe in
``SimulationParams``.

.. literalinclude:: _snippets/compute_projected_area.py
   :language: python

Behavior
========

- ``fl.measure.projected_area(...)`` computes immediately and returns a concrete
  project-length-unit-squared value. It does not modify ``SimulationParams``.
- Assigning ``fl.ProjectedArea(...)`` to ``ReferenceGeometry.area`` records the
  automatic recipe. Python and the Web user interface recompute its output-only
  ``computed`` value before submission.
- Submission must occur while the Geometry-root draft context that owns the
  selected surfaces is active. Surface-mesh and volume-mesh drafts do not carry
  the required tessellation data.
- The uploaded simulation JSON retains both the recipe and the latest
  ``computed`` value, allowing automatic behavior to survive Web user interface
  and Python round trips.
- Coordinate-system rotation and scale are applied before projection.

Approximation
=============

The calculation rasterizes the union of projected triangle coverage. Overlapping
front and back surfaces are counted once. Projected bounds define the raster
frame; their bounding-box area is not used as the result.

Because whole pixels are counted, the result carries a discretization error
proportional to pixel size, with an essentially random sign rather than a
consistent bias. It is largest for small features measured inside a large
bounding box, and for straight edges that happen to fall between pixel centres;
curved silhouettes average out considerably better. At the default settings
expect on the order of 0.1%, so treat a difference of that size between two
measurements of the same geometry as expected rather than as a defect.

New Python code should use ``ProjectedArea`` for automatic behavior and must not
write the legacy Web user interface field ``private_attribute_area_settings``.

Half-body domains
=================

With ``domain_type="half_body_positive_y"`` or ``"half_body_negative_y"``, the
recipe measures only the half that is actually meshed, trimming the tessellation
at ``Y=0``. Do not apply a ``0.5`` factor of your own: the trim is derived from
the finalized meshing settings, so it is already correct whether the uploaded
geometry is a full model or a single half. Projection along ``Y`` is unaffected.

``fl.measure.projected_area`` has no such awareness -- it measures the surfaces as
given -- which is the main reason to prefer the recipe for a reference area.

Confirming the selection
========================

Selectors that match nothing raise no error, so a missed surface shows up only as
a reference area that is quietly too small. ``draft.preview_unselected(...)``
lists every surface the selection does not cover, which makes the omission
visible before submission:

.. code-block:: python

    missed = draft.preview_unselected(projected_area_surfaces)
    if missed:
        raise ValueError(f"No selector covers: {missed}")

Surfaces intentionally excluded -- wind tunnel walls, for example -- appear in
that list and are expected there.

.. seealso::

   - :doc:`Reference Dimensions </gui_guide/02.simulation-setup/04.output/reference-dimensions>`
   - :doc:`Measurement API <../API_reference/measurement>`
   - :doc:`Reference Geometry API <../API_reference/reference_geometry>`
   - :ref:`Draft: auditing a selection <python_api_draft>`
   - :ref:`Asset Drafts user guide <asset_drafts_userGuide>`
