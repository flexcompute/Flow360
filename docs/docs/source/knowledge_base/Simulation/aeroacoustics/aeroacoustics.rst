.. _aeroacousticsCapabilities:

Aeroacoustics
========================================

Overview
--------

Flow360 provides the ability to estimate noise spectrum and level due to aerodynamics using an F1A-based Ffowcs Williams-Hawkings (FWH) aeroacoustics solver.
This allows for the prediction of the acoustics pressure time history from acoustic sources that are primarily derived due to aerodynamical pressure fluctuations
on solid surfaces. Flow360's aeroacoustics solver is source time-based, which results in higher accuracy and lower execution time as compared to observer time-based formulations.

The FWH equations model is an inhomogeneous wave equation that can be derived from the Navier-Stokes equations by Lighthill's analogy. It contains many source terms that
can contribute to the production of noise: monopole, dipole, and quadrupole. The first two terms are computed by taking integrals over a solid surface and the last term is a volumetric integral.
In Flow360’s aeroacoustic solver, we don't consider the contribution from quadrupole terms as it greatly increases the computational cost of the aeroacoustics solver. We only use the first two terms to estimate the amplitude of the acoustics signal at observer locations and our testing has shown that it is sufficient to accurately predict the sound pressure profile for a majority of aeroacoustics applications.

Flow360's aeroacoustics solver supports a single SymmetryPlane boundary to mirror the acoustic signals for all no-slip wall boundaries.
