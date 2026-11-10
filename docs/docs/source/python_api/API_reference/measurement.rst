Measurement
-----------

.. currentmodule:: flow360.component.simulation.measurement

.. autofunction:: projected_area

.. autofunction:: oriented_bounding_box

.. currentmodule:: flow360

``projected_area`` returns a concrete value immediately, which makes it useful for
inspecting a geometry or comparing candidate surface sets. The result is the
silhouette union of the selected surfaces: overlapping surfaces are counted once,
and duplicates in the input are ignored, so there is no need to deduplicate
first. It accepts a single ``Surface``, a single ``SurfaceSelector``, a list
mixing the two, or an ``EntityList[Surface]``.

For a **reference area**, prefer the recipe: assign :class:`ProjectedArea` to
``ReferenceGeometry.area`` instead of computing a number and storing it. The
recipe keeps the surface selection in the uploaded ``simulation.json``, and it is
the only path that accounts for half-body meshing automatically -- see
:doc:`Reference Geometry <reference_geometry>`.

The result is rasterized rather than integrated exactly, so it carries a small
discretization error -- on the order of 0.1% at the default settings.

.. admonition:: Half-body domains
   :class: warning

   ``projected_area`` measures exactly the surfaces it is given and knows nothing
   about the meshing settings. With a ``half_body_*`` domain its result therefore
   describes the geometry as uploaded, not the half that gets meshed. Use
   :class:`ProjectedArea` rather than scaling the returned value by hand.

``DraftContext.compute_obb()`` remains as a deprecated wrapper around
``measure.oriented_bounding_box()`` and is scheduled for removal in release 26.
