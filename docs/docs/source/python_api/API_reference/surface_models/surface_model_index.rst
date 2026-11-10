
.. _surface_models:

Surface Models
==============
.. currentmodule:: flow360

.. autosummary::
   :toctree: ../_autosummary
   :template: class.rst

   Wall
   SlipWall
   Freestream
   Inflow
   Outflow
   Periodic
   SymmetryPlane
   PorousJump
   

Surface Model Specifications
==================================

.. grid:: 2

    .. grid-item-card:: Surface Model Specs
        :link: surface_model_specs
        :link-type: doc
        
        Specification classes used by surface models (e.g., TotalPressure, MassFlowRate, Supersonic, Pressure, Mach, etc.)

    .. grid-item-card:: Surface Model Functions
        :link: surface_model_functions
        :link-type: doc
        
        Utility functions and classes for surface models (e.g., TurbulenceQuantities)

.. toctree::
   :hidden:

   surface_model_specs
   surface_model_functions
