.. _python_api_compute_obb:

.. currentmodule:: flow360

******************************************
Estimate a wheel rotation axis with an OBB
******************************************

This example shows how to use the draft context to fit an oriented bounding box
(OBB) to a set of wheel surfaces, then derive the rotation axis and radius of the
wheel from that box. These values can be used to configure a rotating wall
boundary condition or a rotating reference frame without measuring the geometry
by hand.

.. literalinclude:: _snippets/compute_obb.py
   :language: python

Notes
=====

- ``compute_obb`` is available only for drafts created from a **Geometry** asset. Drafts created from a surface mesh or volume mesh do not carry the tessellation data required to fit the box.
- ``entities`` accepts a single surface, a list of surfaces, or a surface view (e.g. ``draft.surfaces[...]``). Non-surface entities (for example mirrored surfaces) are skipped with a warning.
- ``compute_obb`` returns an ``OBBResult`` that describes the box geometry only: ``center``, ``axes`` (the principal axes as row vectors) and ``extents`` (the half-extents along each axis). When the project has a length unit, ``center`` and ``extents`` carry that unit.
- The rotation axis and radius are obtained separately via ``OBBResult.get_rotation_axis_and_radius()``, which returns a ``RotationAxisAndRadius`` with ``axis_of_rotation`` (a unit vector) and ``averaged_radius``. The averaged radius is the mean of the two box half-extents perpendicular to the rotation axis, and inherits the project length unit from those ``extents``. Printing the ``RotationAxisAndRadius`` shows the values that were averaged.
- Select the rotation axis with **either** ``rotation_axis_hint`` (the principal axis most aligned with the given direction is used) **or** ``axis_index`` (0, 1 or 2). Passing both, an out-of-range index, or a zero hint raises ``Flow360ValueError``. If neither is given, the axis is inferred from the most circular cross-section and a warning is emitted, since this is an inference rather than a known geometric property.

Example use cases
=================

- Automatic rotation setup for wheels and other cylindrical components
- Estimating component size and orientation directly from geometry

.. seealso::

   - :ref:`Draft API Reference <python_api_draft>` (details of ``DraftContext.compute_obb`` and ``OBBResult``)
   - :ref:`Asset Drafts user guide <asset_drafts_userGuide>` (conceptual overview)
