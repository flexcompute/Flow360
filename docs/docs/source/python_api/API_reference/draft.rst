.. _python_api_draft:

*****
Draft
*****


A draft is an isolated, in-memory snapshot of an asset's entity information. It lets you inspect and modify entities (surfaces, edges, volumes, body groups, etc.) locally without mutating the cloud asset.

.. currentmodule:: flow360.component.simulation.draft_context

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   DraftContext


The draft is created using the `create_draft()` function. It is meant to be used with the `with` statement to create a context manager.

.. currentmodule:: flow360

.. code-block:: python

    with fl.create_draft(new_run_from=geometry) as draft:
        ...

.. autosummary::
   :toctree: _autosummary
   :template: function.rst

   create_draft

The draft is responsible primarily for the following:

- grouping **faces**, **edges** and **body groups**
- keeping track of all the existen entities
- managing **mirror** actions
- managing **coordinate systems**
- analyzing geometry through the **oriented bounding box**

Access properties
-----------------

The properties of the :class:`~flow360.component.simulation.draft_context.DraftContext` that store the geometric entities within the draft are listed below. 
The entities within those properties can be accessed by name or pattern with the only exception being the imported geometries and surfaces, which can be accessed by index.

.. currentmodule:: flow360.component.simulation.draft_context

.. autosummary::

   DraftContext.body_groups
   DraftContext.surfaces
   DraftContext.mirrored_body_groups
   DraftContext.mirrored_surfaces
   DraftContext.edges
   DraftContext.volumes
   DraftContext.boxes
   DraftContext.cylinders
   DraftContext.imported_geometries
   DraftContext.imported_surfaces

Management properties
---------------------

Managing **mirror** actions and **coordinate systems** is done through the following properties.

.. autosummary::

   DraftContext.coordinate_systems
   DraftContext.mirror


Those properties provide access to relevant managers. 

.. currentmodule:: flow360.component.simulation.draft_context

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   ~coordinate_system_manager.CoordinateSystemManager
   ~mirror.MirrorManager
   
Actions and objects registered through those managers can be modified by directly accessing them through the manager objects.

**Example**: geometric parameter sensitivity study

.. code-block:: python

    with draft:
        draft.coordinate_systems.assign(
            entities=draft.body_groups["body_group_1"],
            coordinate_system=fl.CoordinateSystem(
                name="body_group_1_translation",
                origin=[0, 0, 0] * fl.u.mm,
                axis_of_rotation=(0, 0, 1),
                angle_of_rotation=0 * fl.u.deg,
                scale=(1.0, 1.0, 1.0),
                translation=[20, 0, 0] * fl.u.mm
            )
        )

        prj.run_case(...)

        draft.coordinate_systems.get_by_name("body_group_1_translation").angle_of_rotation = 5*fl.u.deg

        prj.run_case(...)

.. admonition:: Important
   :class: danger

   Mirroring and custom coordinate systems are available only with GeometryAI enabled.

Auditing a selection
--------------------

Selectors fail quietly. A glob that matches nothing, or a name whose
capitalization does not match the geometry, raises no error -- the surfaces
simply never appear in any assignment, and the first sign of trouble is a
reference area or a boundary condition that covers less than intended. Two
draft methods make the selection visible before submission.

.. currentmodule:: flow360.component.simulation.draft_context

.. autosummary::

   DraftContext.preview_selector
   DraftContext.preview_unselected

``preview_selector`` answers "what does this selector match?" for one selector.
``preview_unselected`` answers the opposite and more useful question -- "what did
I miss?" -- by reporting every entity in the draft that the given selection does
**not** cover.

.. code-block:: python

    with fl.create_draft(new_run_from=geometry, face_grouping="faceId") as draft:
        wheels = fl.SurfaceSelector(name="wheels").match("*rim*")
        body = fl.SurfaceSelector(name="body").match("*body*")

        print(draft.preview_selector(wheels))          # ['front_left_rim', ...]

        missed = draft.preview_unselected([wheels, body])
        if missed:
            raise ValueError(f"No selector covers: {missed}")

The comparison pool is every entity the selection's kind of selector can reach,
read from the same configuration selector expansion itself uses, so the two
cannot drift apart. For surfaces that means ``Surface`` and ``MirroredSurface``;
imported surfaces and ghost boundaries are not selectable and are never
reported. The pool spans the whole draft, so surfaces you deliberately left out
-- wind tunnel walls, for instance -- appear in the result and are expected
there.

Both accept the same selection forms: a single entity, a single selector, a list
mixing the two, or an ``EntityList`` (for example ``my_wall.entities``).
``preview_unselected`` infers the entity kind from the selection, so the
selection must name exactly one kind and must not be empty. By default both
return names; pass ``return_names=False`` for entity instances.

.. admonition:: Note
   :class: note

   Both methods are beta and may change in future releases. Each emits a notice
   once per draft rather than once per call, so they are safe to use in loops.

Computing an oriented bounding box
----------------------------------

For drafts created from a **Geometry** asset, ``fl.measure`` can fit an oriented
bounding box (OBB) to a set of surfaces using the underlying tessellation data.
The box is aligned with the natural axes of the selected surfaces (via principal
component analysis) rather than the global coordinate axes, and gives the
center, principal axes and extents of cylindrical components such as wheels.
From that box a rotation axis and radius can be derived to drive a rotating wall
boundary condition or a rotating reference frame.

.. currentmodule:: flow360

.. autosummary::

   measure.oriented_bounding_box

``oriented_bounding_box`` returns an ``OBBResult`` describing the box geometry only:
``center``, ``axes`` and ``extents``. The rotation axis and radius are obtained
separately through ``OBBResult.get_rotation_axis_and_radius()``, which returns a
``RotationAxisAndRadius``. The axis is selected with either a ``rotation_axis_hint``
direction or an explicit ``axis_index`` (0, 1 or 2). Passing both, an
out-of-range index, or a zero hint raises ``Flow360ValueError``; if neither is
given the axis is inferred from the most circular cross-section and a warning is
emitted.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   ~component.simulation.measurement.bounding_box.OBBResult
   ~component.simulation.measurement.bounding_box.RotationAxisAndRadius

.. admonition:: Note
   :class: note

   ``fl.measure.oriented_bounding_box`` requires a draft created from a Geometry resource. Drafts
   created from a surface mesh or volume mesh do not carry the tessellation
   data the bounding box is computed from.

``DraftContext.compute_obb()`` is a deprecated wrapper around
``fl.measure.oriented_bounding_box()`` and is scheduled for removal in release
26.

.. seealso::

  - :doc:`Windsor body with GeometryAI <../example_library/notebooks/windsor_body>`
  - :doc:`Estimate a wheel rotation axis with an OBB <../grab_and_go_snippets/compute_obb>`
  - :ref:`Asset Drafts user guide <asset_drafts_userGuide>` (conceptual overview of drafts and geometry analysis)
