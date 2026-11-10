release-24.11
=============

Released: 14 November 2024

A new version of Flow360, release-24.11, has been deployed. Any
new submissions of mesh will use this new version by default. Any
cases based on previously submitted meshes or forked from submitted
cases using prior versions will still use their originally specified
versions of Flow360.
   
Solver
------

*New features / Improvements*

1. Expose SA/SST turbulence model coefficients to user

2. User-specification of trip location for transition modelling

3. More accurate prediction of separation with wall function

4. Multiple volume zone support across a wall boundary

5. More accurate and more performance-consistent interpolation for multi-zone rotating mesh

6. Low-Mach Preconditioner for steady/unsteady cases

7. Accelerated unsteady adaptive CFL

8. Linear Solver speed improvement

9. Wall Model Support for Moving wall boundaries

10. Mesh Processor Improvement to handle lower quality elements

11. Added heat source and mesh volume zone support for porous media

12. Improved convergence of transonic cases

13. Added check for correctness of UGRID endianness

*Bugfixes*

1. Bugfix for reporting min/max in SST and AFT solvers

2. Bugfix in restart of AFT solver

3. Bugfix for multiple MassInflow/MassOutflow Boundary Conditions

4. Fixed oscillating residual of kOmegaSST

5. Fixed inaccuracy in estimation of thermal conductivity

6. Bugfix for when slidingInteface intersects with wallFunction boundary

Mesher
------

1. Added a beta version of a fast and robust volume mesher for aerospace and automotive applications

2. Added support for symmetry planes and user defined farfield surfaces in beta mesher

3. Added support for user-defined volume refinement zones in beta mesher

Postprocessing 
--------------

*New features / Improvements*

1. Support for Surface Monitors

2. Line Monitors for both surface and volume

3. Added heatTransferCoefficient, localCFL, and totalPressureCoefficient to the list of supported outputs

4. Restart support for time-averaged volume and surface outputs

5. Time-averaging support for slices

6. Time-averaging support for monitors

*Bugfixes*

1. Fixed a bug which showed non-zero Cf values on boundaries other than noSlipWall

2. Fixed holes in slices/isosurfaces due to non-planar quads

3. Fixed BET radial force distribution csv

Python Client
-------------

*Breaking Changes*

1. A new Python package ``flow360`` has been introduced, while ``flow360client`` is now deprecated. The ``flow360client`` package will continue to work but cannot be used to submit cases to Flow360 version 24.11 or newer.

   To install the new package:

   .. code-block:: bash

      pip install "flow360==24.11.*"

   This will automatically select the latest patch version of 24.11.