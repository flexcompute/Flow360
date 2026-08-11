.. _capabilities_overview:

Capabilities
============

.. rubric:: Meshing

.. list-table::
   :class: capabilities-table
   :widths: 88 12

   * - Automatic surface and volume mesh generation
     - :doc:`GUI </gui_guide/02.simulation-setup/02.mesh/README>` | :ref:`API <python_api_meshing>`
   * - Refinement zones
     - :doc:`GUI </gui_guide/02.simulation-setup/02.mesh/06.refinements/README>` | :ref:`API <python_api_meshing>`
   * - Rotor disk meshing
     - :doc:`GUI </gui_guide/02.simulation-setup/02.mesh/03.rotation-zones>` | :class:`API <flow360.AxisymmetricRefinement>`
   * - Sliding interfaces
     - :doc:`GUI </gui_guide/02.simulation-setup/02.mesh/03.rotation-zones>` | :class:`API <flow360.RotationVolume>`

.. rubric:: Geometry cleanup and manipulation

.. list-table::
   :class: capabilities-table
   :widths: 88 12

   * - GeometryAI surface meshing
     - :ref:`User guide <GAI_surface_mesher_user_guide>` | :doc:`Python example </python_api/example_library/notebooks/windsor_body>`
   * - CAD metadata ingestion and face grouping
     - :doc:`GUI </gui_guide/04.entities-browser/01.geometry/02.faces>` | :class:`API <flow360.Geometry.group_faces_by_tag>`

.. rubric:: Modeling options

.. list-table::
   :class: capabilities-table
   :widths: 88 12

   * - Steady and unsteady viscous flows
     - :doc:`GUI </gui_guide/02.simulation-setup/03.flow-solver/02.time>` | :ref:`API <timeStepping>`
   * - Coupled stationary and rotating domains
     - :doc:`GUI </gui_guide/02.simulation-setup/03.flow-solver/03.physics/03.rotation>` | :class:`API <flow360.Rotation>`
   * - Reynolds-Averaged Navier-Stokes
     - :doc:`GUI </gui_guide/02.simulation-setup/03.flow-solver/03.physics/01.fluid/01.navier-stokes-solver>` | :class:`API <flow360.NavierStokesSolver>`
   * - Actuator disk
     - :doc:`GUI </gui_guide/02.simulation-setup/03.flow-solver/03.physics/05.actuator-disk>` | :class:`API <flow360.ActuatorDisk>`
   * - Blade element theory (BET)
     - :doc:`GUI </gui_guide/02.simulation-setup/03.flow-solver/03.physics/04.bet-disk>` | :class:`API <flow360.BETDisk>`
   * - Porous media
     - :doc:`GUI </gui_guide/02.simulation-setup/03.flow-solver/03.physics/06.porous-medium>` | :class:`API <flow360.PorousMedium>`
   * - Conjugate heat transfer
     - :doc:`GUI </gui_guide/02.simulation-setup/03.flow-solver/03.physics/02.solid/01.heat-equation-solver>` | :class:`API <flow360.HeatEquationSolver>`
   * - Aeroacoustics simulation
     - :doc:`GUI </gui_guide/02.simulation-setup/04.output/02.outputs-list/13.aeroacoustic-output>` | :class:`API <flow360.AeroAcousticOutput>`

.. rubric:: Turbulence modelling

.. list-table::
   :class: capabilities-table
   :widths: 88 12

   * - Spalart-Allmaras (SA-neg)
     - :doc:`GUI </gui_guide/02.simulation-setup/03.flow-solver/03.physics/01.fluid/02.turbulence-model>` | :class:`API <flow360.SpalartAllmaras>`
   * - Spalart-Allmaras with rotation-curvature correction (SA-RC)
     - :doc:`GUI </gui_guide/02.simulation-setup/03.flow-solver/03.physics/01.fluid/02.turbulence-model>` | :class:`API <flow360.SpalartAllmaras>`
   * - Spalart-Allmaras with quadratic constitutive relation (SA-QCR)
     - :doc:`GUI </gui_guide/02.simulation-setup/03.flow-solver/03.physics/01.fluid/02.turbulence-model>` | :class:`API <flow360.SpalartAllmaras>`
   * - k-:math:`{\omega}` SST
     - :doc:`GUI </gui_guide/02.simulation-setup/03.flow-solver/03.physics/01.fluid/02.turbulence-model>` | :class:`API <flow360.KOmegaSST>`
   * - k-:math:`{\omega}` SST with quadratic constitutive relation (SST-QCR)
     - :doc:`GUI </gui_guide/02.simulation-setup/03.flow-solver/03.physics/01.fluid/02.turbulence-model>` | :class:`API <flow360.KOmegaSST>`
   * - Delayed Detached Eddy Simulation (DDES)
     - :doc:`GUI </gui_guide/02.simulation-setup/03.flow-solver/03.physics/01.fluid/02.turbulence-model>` | :class:`API <flow360.DetachedEddySimulation>`
   * - Zonal Detached Eddy Simulation (ZDES)
     - :doc:`GUI </gui_guide/02.simulation-setup/03.flow-solver/03.physics/01.fluid/02.turbulence-model>` | :class:`API <flow360.DetachedEddySimulation>`
   * - Amplification factor transport (AFT), transition model
     - :doc:`GUI </gui_guide/02.simulation-setup/03.flow-solver/03.physics/01.fluid/03.transition-model>` | :class:`API <flow360.TransitionModelSolver>`
   * - Shear-layer-adapted (SLA) grid scale for DES
     - :doc:`GUI </gui_guide/02.simulation-setup/03.flow-solver/03.physics/01.fluid/02.turbulence-model>` | :class:`API <flow360.DetachedEddySimulation.grid_size_for_LES>`
.. rubric:: Boundary conditions

.. list-table::
   :class: capabilities-table
   :widths: 88 12

   * - Freestream
     - :doc:`GUI </gui_guide/02.simulation-setup/03.flow-solver/01.boundary-conditions/02.freestream>` | :class:`API <flow360.Freestream>`
   * - Slip wall
     - :doc:`GUI </gui_guide/02.simulation-setup/03.flow-solver/01.boundary-conditions/07.slip-wall>` | :class:`API <flow360.SlipWall>`
   * - Symmetry plane
     - :doc:`GUI </gui_guide/02.simulation-setup/03.flow-solver/01.boundary-conditions/06.symmetry>` | :class:`API <flow360.SymmetryPlane>`
   * - Wall (no-slip, isothermal, heat flux, wall model)
     - :doc:`GUI </gui_guide/02.simulation-setup/03.flow-solver/01.boundary-conditions/01.wall>` | :class:`API <flow360.Wall>`
   * - Inflow (subsonic inflow, mass flow in)
     - :doc:`GUI </gui_guide/02.simulation-setup/03.flow-solver/01.boundary-conditions/03.inflow>` | :class:`API <flow360.Inflow>`
   * - Outflow (back pressure, Mach, mass flow out)
     - :doc:`GUI </gui_guide/02.simulation-setup/03.flow-solver/01.boundary-conditions/04.outflow>` | :class:`API <flow360.Outflow>`
   * - Periodic (translational/rotational)
     - :doc:`GUI </gui_guide/02.simulation-setup/03.flow-solver/01.boundary-conditions/05.periodic>` | :class:`API <flow360.Periodic>`
   * - Porous jump
     - :doc:`GUI </gui_guide/02.simulation-setup/03.flow-solver/01.boundary-conditions/08.porous-jump>` | :class:`API <flow360.PorousJump>`

.. rubric:: Post processing and visualization

.. list-table::
   :class: capabilities-table
   :widths: 88 12

   * - Simulation outputs (volume, surface, slice, probe, isosurface, force, aeroacoustic)
     - :doc:`GUI </gui_guide/02.simulation-setup/04.output/README>` | :ref:`API <outputs>`
   * - 3D visualization
     - :doc:`GUI </gui_guide/03.analysis/04.visualization>`
   * - Surface mesh diagnostics
     - :doc:`GUI </gui_guide/05.tools/03.surface-mesh-diagnostics>`
   * - Volume mesh diagnostics
     - :doc:`GUI </gui_guide/05.tools/04.volume-mesh-diagnostics>`
   * - Volume mesh slices
     - :doc:`GUI </gui_guide/02.simulation-setup/02.mesh/05.volume-mesh-slices>` | :class:`API <flow360.MeshSliceOutput>`
   * - Streamlines
     - :doc:`GUI </gui_guide/02.simulation-setup/04.output/02.outputs-list/14.streamline-output>` | :class:`API <flow360.StreamlineOutput>`

.. rubric:: User customization

.. list-table::
   :class: capabilities-table
   :widths: 88 12

   * - User-defined dynamics
     - :doc:`GUI </gui_guide/02.simulation-setup/03.flow-solver/04.user-defined-dynamics>` | :ref:`User guide <user_defined_dynamic_user_guide>` | :ref:`API <user_defined_dynamics>`
   * - Custom simulation variables and post-processing expressions
     - :doc:`GUI: </gui_guide/05.tools/02.variable-settings>` | :ref:`User guide <user_defined_expressions_user_guide>` | :ref:`API <api_userDefinedExpressions>`
   * - User-defined stopping criteria
     - :doc:`GUI </gui_guide/02.simulation-setup/03.flow-solver/03.physics/01.fluid/05.stopping-criteria>` | :ref:`User guide <runControl>` | :ref:`API <python_api_run_control>`
