release-23.3.2.0
================

Released: 26th October 2023

A new version of Flow360, release-23.3.2.0, has been deployed. Any
new submissions of mesh will use this new version by default. Any
cases based on previously submitted meshes or forked from submitted
cases using prior versions will still use their originally specified
versions of Flow360.
   
Solver
------

*New features*

1. Arbitrary non-conformal rotating volume zones with no concentric ring requirement on sliding interface.

2. Added Volume Zone Grid Connectivity based CGNS file support for multi zone physics simulations.

3. Added support for unsteady conjugate heat transfer.

4. Added support for rotationally periodic boundary condition.

5. Added multiple reference frame (MRF) feature for steady state simulations of multiple rotating volume zones.

5. Added single reference frame (SRF) feature for steady state simulations of single rotating volume zones.

6. Added aeroacoustics solver for noise prediction based on FWH method.

*Resolved issues*

1. Fixed minimum and maximum of k and omega solution for kOmegaSST turbulence model in solver log.

2. Unified volumetric output of temperature including fluid and solid zones for conjugate heat transfer simulations.

3. Added validation for requiring muRef when Mach=0.

4. Improved robustness and accuracy of isothermal boundary condition.

5. Improved accuracy of turbulence models inside rotating volume zones.

6. Improved robustness of SpalartAllmaras turbulence model and fixed convergence issues when coupled with transition model.

flow360client
-------------

Although recent versions of flow360client will still work, it is
highly recommended to upgrade to the latest version for more
convenient capabilities:

Usage:

- If downloading for the first time: :code:`pip3 install flow360client`

- If upgrading from an older version: :code:`pip3 install --upgrade flow360client`

Here is the flow360client on PyPI\: https://pypi.org/project/flow360client/
