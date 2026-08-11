.. _cloud_assets:

Cloud Assets
------------
.. currentmodule:: flow360

Flow360 cloud assets represent the core resources in a CFD simulation workflow. These assets are organized hierarchically within a :ref:`Project <project_management>` and form the foundation of your CFD analysis, from geometry upload through mesh generation to simulation execution.

For comprehensive information on project structure, asset types, and workflows, see the :ref:`Project Management <project_management>` section in the User Guide.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

    Project

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

    Case
    Geometry
    SurfaceMesh
    VolumeMesh
    ~component.surface_mesh_v2.SurfaceMeshStats
