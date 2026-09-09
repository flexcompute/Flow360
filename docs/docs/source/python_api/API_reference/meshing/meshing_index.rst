.. _python_api_meshing:

*******
Meshing
*******

.. currentmodule:: flow360

Meshing Interfaces
==================

.. autosummary::
   :toctree: ../_autosummary
   :template: class.rst

   MeshingParams
   ModularMeshingWorkflow


Unified Meshing API
===================

Classes related to the :class:`MeshingParams` interface.

.. autosummary::
   :toctree: ../_autosummary
   :template: class.rst

   MeshingDefaults

   SurfaceEdgeRefinement
   AngleBasedRefinement
   HeightBasedRefinement
   AspectRatioBasedRefinement
   ProjectAnisoSpacing
   
   SurfaceRefinement

   BoundaryLayer
   PassiveSpacing
   UniformRefinement
   AxisymmetricRefinement
   StructuredBoxRefinement
   GeometryRefinement

Modular Meshing Workflow
========================

Classes related to the :class:`ModularMeshingWorkflow` interface.

Snappy Mesher
-------------

.. autosummary::
   :toctree: ../_autosummary
   :template: class.rst

   snappy.SurfaceMeshingParams

Specifications
^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ../_autosummary
   :template: class.rst

   snappy.CastellatedMeshControls
   snappy.QualityMetrics
   snappy.SmoothControls
   snappy.SnapControls
   snappy.SurfaceMeshingDefaults
   OctreeSpacing

Refinements
^^^^^^^^^^^

.. autosummary::
   :toctree: ../_autosummary
   :template: class.rst

   snappy.BodyRefinement
   snappy.RegionRefinement
   snappy.SurfaceEdgeRefinement

.. autosummary::
   :template: class.rst
   
   UniformRefinement


New Volume Mesher
-----------------

.. autosummary::
   :toctree: ../_autosummary
   :template: class.rst

   VolumeMeshingParams
   VolumeMeshingDefaults

.. autosummary::
   :template: class.rst

   UniformRefinement
   BoundaryLayer
   PassiveSpacing
   AxisymmetricRefinement
   StructuredBoxRefinement


Zones
=====

Zones are used to define the creation of volume zones in the meshers, regardless of the interface.

.. autosummary::
   :toctree: ../_autosummary
   :template: class.rst

   AutomatedFarfield
   UserDefinedFarfield
   WindTunnelFarfield
   CustomZones

.. autosummary::
   :template: class.rst

   RotationVolume
   RotationSphere

Outputs
=======

To inspect the generated mesh, specific meshing outputs need to be defined.

.. autosummary::
   :toctree: ../_autosummary
   :template: class.rst

   MeshSliceOutput

Farfield Zone Specifications
=============================

.. grid:: 1

    .. grid-item-card:: Farfield Specs
        :link: farfield_specs
        :link-type: doc
        
        Specification classes for farfield zones (e.g., AutomatedFarfield, UserDefinedFarfield, WindTunnelFarfield)

.. toctree::
   :hidden:

   farfield_specs