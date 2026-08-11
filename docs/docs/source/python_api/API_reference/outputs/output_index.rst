
.. _outputs:

.. currentmodule:: flow360

*******
Outputs
*******

Output Classes
==============

Classes for defining various types of simulation outputs.

.. autosummary::
   :toctree: ../_autosummary
   :template: class.rst

   VolumeOutput
   TimeAverageVolumeOutput
   SliceOutput
   TimeAverageSliceOutput
   SurfaceOutput
   TimeAverageSurfaceOutput
   IsosurfaceOutput
   SurfaceIntegralOutput
   SurfaceSliceOutput
   ProbeOutput
   TimeAverageProbeOutput
   SurfaceProbeOutput
   TimeAverageSurfaceProbeOutput
   ForceOutput
   ForceDistributionOutput
   TimeAverageForceDistributionOutput
   StreamlineOutput
   TimeAverageStreamlineOutput
   RenderOutput
   AeroAcousticOutput

Utility Classes
===============

Helper classes and utilities for configuring outputs.

.. autosummary::
   :toctree: ../_autosummary
   :template: class.rst

   MovingStatistic
   RenderOutputGroup

Render Configuration Classes
============================

Classes for configuring render output materials, cameras, and lighting.

.. currentmodule:: flow360.render_config

.. autosummary::
   :toctree: ../_autosummary
   :template: class.rst

   Camera
   PBRMaterial
   FieldMaterial

Output Configuration
====================

.. grid:: 2

    .. grid-item-card:: Output Fields
        :link: output_fields
        :link-type: doc
        
        Available output fields and variables for different output types

    .. grid-item-card:: Output Entities
        :link: output_entities
        :link-type: doc
        
        Entity selection and configuration for outputs

.. toctree::
   :hidden:

   output_fields
   output_entities








