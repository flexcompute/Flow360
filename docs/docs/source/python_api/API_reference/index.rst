.. _python_API_reference:

.. currentmodule:: flow360

*************
API Reference
*************

Complete reference documentation for the Flow360 Python API. This section provides detailed information about all classes, functions, and modules available in the Flow360 package.

Getting Started
===============

Essential components for initializing and configuring Flow360 simulations.

.. grid:: 2

    .. grid-item-card:: Setup
        :link: setup
        :link-type: doc
        
        Core setup classes including :class:`SimulationParams` for configuring simulation parameters

    .. grid-item-card:: Cloud Assets
        :link: cloud_assets
        :link-type: doc
        
        Classes for managing cloud resources: :class:`Project`, :class:`Case`, :class:`Geometry`, :class:`SurfaceMesh`, :class:`VolumeMesh`

    .. grid-item-card:: Entities
        :link: entities
        :link-type: doc
        
        Geometric entities for defining volumes, surfaces, and points in your simulation domain

    .. grid-item-card:: Draft Context
        :link: draft
        :link-type: doc
        
        Draft context for geometry analysis and surface/edge identification before meshing

Meshing
=======

Configuration and parameters for mesh generation.

.. grid:: 1

    .. grid-item-card:: Meshing
        :link: meshing/meshing_index
        :link-type: doc
        
        Meshing interfaces, refinements, and zone definitions for surface and volume mesh generation

Geometry
========

Geometric reference properties.

.. grid:: 1

    .. grid-item-card:: Reference Geometry
        :link: reference_geometry
        :link-type: doc
        
        Reference geometry definitions for force and moment calculations

Simulation Configuration
=========================

Physical models, boundary conditions, and solver settings for defining the flow physics and numerical methods.

.. grid:: 2

    .. grid-item-card:: Operating Condition
        :link: operating_condition/operating_condition_index
        :link-type: doc
        
        Freestream conditions, reference values, and operating parameters

    .. grid-item-card:: Material
        :link: operating_condition/material
        :link-type: doc
        
        Material properties and fluid models

    .. grid-item-card:: Volume Models
        :link: volume_models/volume_model_index
        :link-type: doc
        
        Volume physics models: :class:`Fluid`, :class:`Solid`, :class:`Rotation`, :class:`BETDisk`, :class:`ActuatorDisk`, :class:`PorousMedium`

    .. grid-item-card:: Surface Models
        :link: surface_models/surface_model_index
        :link-type: doc
        
        Boundary conditions: :class:`Wall`, :class:`Freestream`, :class:`Inflow`, :class:`Outflow`, :class:`Periodic`, :class:`SymmetryPlane`

    .. grid-item-card:: Time Stepping
        :link: time_stepping/time_stepping_index
        :link-type: doc
        
        Time stepping schemes and temporal discretization methods

    .. grid-item-card:: Solver Configuration
        :link: solver_configuration/solver_index
        :link-type: doc
        
        Solver numerics, convergence criteria, and algorithm settings

Advanced Features
=================

Custom expressions, dynamics, and run control.

.. grid:: 3

    .. grid-item-card:: User Defined Expressions
        :link: user_defined_expressions
        :link-type: doc
        
        Custom variables and mathematical expressions for simulation parameters

    .. grid-item-card:: User Defined Dynamics
        :link: user_defined_dynamics
        :link-type: doc
        
        Custom control laws and coupled physics models

    .. grid-item-card:: Run Control
        :link: run_control/run_control_index
        :link-type: doc
        
        Simulation execution control, restart options, and convergence monitoring

Outputs and Results
===================

Configuration for simulation outputs and result analysis.

.. grid:: 2

    .. grid-item-card:: Outputs
        :link: outputs/output_index
        :link-type: doc
        
        Output classes for volume, surface, slice, probe, force, streamline, and aeroacoustic outputs

    .. grid-item-card:: Results
        :link: results
        :link-type: doc
        
        Result processing, interpolation, and data manipulation utilities

Reporting
=========

Report generation and visualization.

.. grid:: 1

    .. grid-item-card:: Report
        :link: report/report_index
        :link-type: doc
        
        Report templates, charts, and data visualization tools

.. toctree::
   :hidden:
   :maxdepth: 1

   setup
   cloud_assets
   entities
   draft
   meshing/meshing_index
   reference_geometry
   operating_condition/operating_condition_index
   volume_models/volume_model_index
   surface_models/surface_model_index
   time_stepping/time_stepping_index
   solver_configuration/solver_index
   run_control/run_control_index
   user_defined_expressions
   user_defined_dynamics
   outputs/output_index
   results
   report/report_index
