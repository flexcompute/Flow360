release-25.5
=============

Released: 15 May 2025

A new version of Flow360, release-25.5, has been deployed. Any
new submissions of mesh will use this new version by default. Any
cases based on previously submitted meshes or forked from submitted
cases using prior versions will still use their originally specified
versions of Flow360.

Solver
------

*New features / Improvements*

1. Added support for liquid simulation (heat transfer not supported)
2. Added detailed timing for various components of Flow360 mesher and solver
3. Improved convergence of Turbulence/Transition solvers, leading to deeper residual convergence
4. Improved performance of Navier-Stokes solver, leading to up to 15% reduction in simulation time

*Bugfixes*

1. Fixed a bug which led to case errors for some simulations with nested rotating zones
2. Fixed a bug which led to zero Cf values on the intersection of `NoSlipWall` and `SymmetryPlane` boundaries

Mesher
------

*New features / Improvements*

1. Added support for processing multiple input CAD and surface mesh files
2. Added multi-zone support for isolated rotating zones with cylindrical interfaces to beta volume mesher
3. Automatic identification and removal of non-manifold edges in beta volume mesher
4. Improved BRep geometry healing capabilities and the propagation of body attributes
5. Improved quality of anisotropic refinement in beta surface mesher
6. Improved the accuracy of mesh sizing function in beta volume mesher
7. Increased robustness of boundary layer growth in beta volume mesher
8. Improved performance of surface remeshing in beta volume mesher
9. Better user logs and error messages in beta volume mesher

Postprocessing 
--------------

*New features / Improvements*

1. Added support for streamlines output
2. Added vorticity magnitude to the list of supported outputs

*Bugfixes*

1. Fixed the performance issue related to cases with volume monitor output

Python Client
-------------

*New features / Improvements*

1. Multi‑variable plotting in Chart2D and new NonlinearResiduals report object
2. Limited forward‑compatibility handshake for Python client versions
3. Water/Liquid‑phase OperatingCondition interface
4. X / Y chart‑limit controls in report generation
5. Solution interpolation onto given mesh when forking a case
6. Allow user to setup APIKey for certain ENV using Python scripts when command line does not work
7. Added kinematic viscosity, fixed turbulent viscosity type
8. Improvements validation and input handling
9. Better error messaging
10. Relaxed duplicate entity checks

*Bugfixes*

1. Reference‑velocity bug in `op_from_mach_reynolds` fixed
2. Incorrect captions in 2‑D charts fixed
3. Removed `sys.stdout` spam in notebooks


To install the new package:
.. code-block:: bash

   pip install "flow360==25.5.*"

This will automatically select the latest patch version of 25.5.


*Documentation & examples*

1. Example sweep template for automated reports
2. Updated 2‑D CRM notebook
3. Re‑structured examples folder for 25.5 layout
4. Two new template example scripts





