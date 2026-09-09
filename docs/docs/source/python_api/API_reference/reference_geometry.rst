
Reference Geometry
------------------
.. currentmodule:: flow360

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   ReferenceGeometry
   ProjectedArea

Automatic projected reference area
----------------------------------

Assigning :class:`ProjectedArea` to ``ReferenceGeometry.area`` records a
*recipe* -- which surfaces and which direction -- instead of a number. The value
is measured at submission and stored alongside the recipe, so the choice of
surfaces survives into the uploaded ``simulation.json`` and remains inspectable
in the web interface afterwards. Assigning a concrete area computed beforehand
stores only the number, and the reasoning behind it is lost.

.. code-block:: python

    with fl.create_draft(new_run_from=geometry, face_grouping="faceId") as draft:
        body = fl.SurfaceSelector(name="body").match("*body*")
        wheels = fl.SurfaceSelector(name="wheels").match("*rim*")

        with fl.SI_unit_system:
            params = fl.SimulationParams(
                reference_geometry=fl.ReferenceGeometry(
                    area=fl.ProjectedArea(
                        surfaces=[body, wheels],
                        direction="X",
                    ),
                ),
                ...
            )

        # Submit inside the draft context: the measurement needs the tessellation.
        project.run_case(params=params)

The result is the **silhouette union** of the selected surfaces. Surfaces that
overlap along the projection direction are counted once, not summed, and a
surface named more than once -- listed twice, or matched by several selectors --
contributes once. There is no need to deduplicate before assigning.

.. admonition:: Half-body domains: do not scale by hand
   :class: danger

   When the meshing settings use ``domain_type="half_body_positive_y"`` or
   ``"half_body_negative_y"``, the mesher builds the model trimmed at ``Y=0``.
   The reference area is measured over that same half automatically, so the
   stored value already describes the model the solver sees.

   Multiplying by ``0.5`` yourself is wrong in both directions: it double-halves
   a geometry that was uploaded as a full model, and it halves a geometry that
   was already just one half. Because the trim is derived from the finalized
   meshing settings, it is also correct either way -- whether the uploaded
   geometry is a full model the mesher will cut, or already a single half.

   Projecting along ``Y`` -- the trim axis itself -- is unaffected, since the cut
   plane is perpendicular to the view.

.. admonition:: Note
   :class: note

   The measurement rasterizes a silhouette and counts whole pixels, so the result
   carries a small discretization error -- on the order of 0.1% at the default
   settings. Treat a difference of that size between two measurements of the same
   geometry as expected rather than as a defect.

.. seealso::

  - :doc:`Measurement API <measurement>` for the immediate, non-recipe measurement
  - :ref:`Auditing a selection <python_api_draft>` to confirm no surface was missed

