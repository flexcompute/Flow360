release-23.2.1.0
================

Released: 27th April 2023

A new version of Flow360, release-23.2.1.0, has been deployed. Any
new submissions of mesh will use this new version by default. Any
cases based on previously submitted meshes or forked from submitted
cases using prior versions will still use their originally specified
versions of Flow360.
   
Solver
------

*New features*

1. Added option for running unsteady simulations with a low numerical dissipation Roe Flux scheme.

2. Added option for automatically setting CFL based on solution convergence.

3. Minor improvements to efficiency of running simulation, uploading and visualizing results.

4. Added option to modify DDES model constants.

5. Added DDES model volumetric outputs for debugging purposes.

6. Improvements to robustness and convergence of wall model.

7. Improvements to convergence of transition model.

8. Added physical time output to Tecplot output files.

*Resolved issues*

1. Fix SpalartAllmaras turbulence model divergence when linear system convergence is poor.

2. Fix transition model inside sliding interfaces.

3. Fix SpalartAllmaras DDES model to include laminar viscosity in shielding function computation.

4. Improvements to logging and case json validation.

5. Fix visualization for cases with spaces in boundary names.

6. Fix missing fragments of mesh visualization in static pictures.


Automated Meshing
-----------------

*New features*

*Changes*

1. Changed height of anisotropic layers grown on surfaces from edges: now anisotropic layers grow till isotropy.

flow360client
-------------

Although recent versions of flow360client will still work, it is
highly recommended to upgrade to the latest version for more
convenient capabilities:

Usage:

- If downloading for the first time: :code:`pip3 install flow360client`

- If upgrading from an older version: :code:`pip3 install --upgrade flow360client`

Here is the flow360client on PyPI\: https://pypi.org/project/flow360client/
