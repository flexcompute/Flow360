release-24.2
============

Released: 12 March 2024

A new version of Flow360, release-24.2, has been deployed. Any
new submissions of mesh will use this new version by default. Any
cases based on previously submitted meshes or forked from submitted
cases using prior versions will still use their originally specified
versions of Flow360.
   
Solver
------

*New features*

1. Added symmetry plane boundary condition

2. Added symmetry plane boundary condition support for aeroacoustics simulations

3. Improved accuracy for aeroacoustic simulations by increasing compute precision

4. Added ability to fork a simulation onto a mesh that differs from the previous simulation's mesh

5. Improved robustness of conjugate heat transfer for low conductivity solids

6. Added many additional turbulence specifications for initial condition and boundary conditions

7. Added splitting of acoustic signal into loading and thickness contributions for aeroacoustics simulations

8. Added heat flux wall boundary condition

*Resolved issues*

1. Removed CfTangent and CfNormal output variables

2. Improved accuracy of gradient of flow solution output variables

3. Added warning when negative volumes are detected in mesh

4. Improved accuracy of heat flux calculations

5. Improved convergence and accuracy of moving wall simulations

flow360client
-------------

Although recent versions of flow360client will still work, it is
highly recommended to upgrade to the latest version for more
convenient capabilities:

Usage:

- If downloading for the first time: :code:`pip3 install flow360client`

- If upgrading from an older version: :code:`pip3 install --upgrade flow360client`

Here is the flow360client on PyPI\: https://pypi.org/project/flow360client/
