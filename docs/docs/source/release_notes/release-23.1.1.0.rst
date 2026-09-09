release-23.1.1.0
================

Released: 17th February 2023

A new version of Flow360, release-23.1.1.0, has been deployed. Any
new submissions of mesh will use this new version by default. Any
cases based on previously submitted meshes or forked from submitted
cases using prior versions will still use their originally specified
versions of Flow360.
   
Solver
------

*New features*

1. Added support for mesh sizes larger than 250 million nodes.

2. Significantly improved speed of mesh processing.

3. Improved accuracy of gradient computation resulting in improvements to spatial discretization accuracy.

4. Added physical time in Tecplot output for animations.

5. Additional mesh information available in logs.

6. Improved validation messaging for diagnosing issues in Case JSON file.

*Resolved issues*

1. Improved residual convergence of transition model: Amplication Factor Transport.

2. Fixed holes in slice output for certain meshes.

3. Fixed divergence of Spalart Allmaras model with no wall boundaries.

4. Fix relative residual convergence check when initial residual is very small.

5. Reduces discontuity in postprocessing outputs across sliding interface.


Automated Meshing
-----------------

*New features*

1. Improved meshing for C0 continuity edges.

*Changes*

1. Backend uses ESP 1.21 July 2022 release with Open CASCADE Technology 7.4.1


flow360client
-------------

Although recent versions of flow360client will still work, it is
highly recommended to upgrade to the latest version for more
convenient capabilities:

Usage:

- If downloading for the first time: :code:`pip3 install flow360client`

- If upgrading from an older version: :code:`pip3 install --upgrade flow360client`

Here is the flow360client on PyPI\: https://pypi.org/project/flow360client/
