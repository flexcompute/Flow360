release-25.6
=============

Released: 1 August 2025

A new version of Flow360, release-25.6, has been deployed. Any
new submissions of mesh will use this new version by default. Any
cases based on previously submitted meshes or forked from submitted
cases using prior versions will still use their originally specified
versions of Flow360.

Solver
------

*New features / Improvements*

1. Added support for zonal modification of turbulence solver constants, and zonal specification of RANS/DDES/ZDES
2. Improved performance of cases using low mach preconditioner
3. Improved performance of Navier-Stokes, SST, and AFT linear solvers
4. Improved robustness of massflow boundary conditions
5. Reduced turbulence solution discontinuity across rotation zone interfaces
6. Improved stability of the SA solver at freestream boundaries
7. Reduced the memory consumption of the aeroacoustics solver
8. Added support for isosurface of time-averaged quantities

*Bugfixes*

1. Fixed the memory estimation for cases with isosurface outputs

Geometry and Meshing
--------------------

*In-house Surface mesher:*

1. Improved control of surface mesh spacing based on the surface growth rate parameter

*In-house volume mesher:*

2. Added support for symmetry planes with the automated farfield option (also supports input surface meshes that are open at the symmetry plane)
3. Added support for cylindrical BET disks
4. Added check for non-manifold nodes and improved error messages for invalid surface meshes
5. Added support for reading ASCII STL files as a surface mesh input
6. Bug fix for boundary layer edge-splitting

Python Client
-------------

*New features / Improvements*

1. Allow users to supply BETDisk names for migration of BETDisk from v1 Flow360.json
2. Input from Mach-Reynolds now uses a characteristic length
3. Streamlined sweep template report generation
4. Added grouping options for line plots in the automated report generation
5. Added customization to the results summary naming in the automated report generation
6. Support for setting up dimensioned output with user variables
7. Enables setting up fields with user defined expressions including reference area, angular velocity, velocity magnitude, isosurface output value and unsteady time step size
8. Added ``flow360 version`` command for printing latest and installed versions

Web UI
------

*Expression Support & Units System*

1. Define user-defined expressions in the WebUI, which can be used for simulation settings and output settings
2. Define user-defined fields to allow users to generate postprocessing fields of variables based on equations using internal solver variables
3. Visualize user-defined fields using physical units

*Rotating Systems*

1. Automatic setup for rotating wall boundary conditions
2. Visualize rotation direction and axis of rotation for rotating volumes directly in the Web UI

*Visualization Enhancements*

1. Introduce the LOD (Level of Detail) method to support High and Low resolution rendering modes
2. Save high-resolution images directly from the workbench view area

*Workflow & Simulation Tools*

1. Support for porous medium force plots in the UI
2. Added batch delete for run drafts
3. New Run Summary view showing draft models and key values

*Improvements*

1. Prevented duplicate entity selection for boundary conditions
2. All generated volumes and slices now have unique names by default
3. Added planar face (e.g. symmetry plane) tolerance support for beta mesher
4. Visualization tools now exclude hidden entities from "Fit Selected"
5. Improved zoom sensitivity for trackpad-only users
6. Added input fields for axis and center of rotation in volume-mesh-based projects
7. Improved 3D object highlighting and selection to better match the right-hand entity list

To install the new package:

.. code-block:: bash

   pip install "flow360==25.6.*"

This will automatically select the latest patch version of 25.6.


*Documentation & examples*

1. Added Python versions of the isolated propeller and BET eVTOL examples.
2. Added time-averaged isosurface example.
3. Added example showcasing how to obtain FlexCredits used in a project.
