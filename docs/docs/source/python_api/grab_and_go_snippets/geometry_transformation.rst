.. _python_api_coordinate_system_assignment:

.. currentmodule:: flow360

****************************************************
Transform geometric entities with coordinate systems
****************************************************

This example demonstrates how to assign and modify coordinate systems using the draft context manager.
It shows how to attach a coordinate system to a body group, run a case, then modify the coordinate system parameters and run another case.

.. literalinclude:: _snippets/geometry_transformation.py
   :language: python

Notes
=====

- Use ``fl.create_draft(new_run_from=...)`` to create a draft context from a cloud asset (geometry, surface mesh, or volume mesh).
- The ``face_grouping`` and ``edge_grouping`` parameters specify which grouping tags to use for entities (must match tags available on the geometry).
- Use ``coordinate_systems.assign`` to attach a coordinate system to geometry entities (e.g., body groups).
- Coordinate system parameters such as ``angle_of_rotation``, ``translation``, and ``scale`` can be modified between runs to explore different configurations.
- The ``get_by_name`` method retrieves an existing coordinate system for modification.

Example use cases
=================

- Geometric parameter sensitivity studies (rotation, translation, scaling)
- Positioning components at different locations
- Orientation sweeps for aerodynamic analysis
