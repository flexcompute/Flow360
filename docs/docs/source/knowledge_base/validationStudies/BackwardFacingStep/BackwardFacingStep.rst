.. _backward_facing_step:
.. |agr|  unicode:: U+003B1 .. GREEK SMALL LETTER ALPHA
.. |omega|    unicode:: U+03C9 .. OMEGA SIGN


2D Backward Facing Step
=============================================

Introduction
------------

The main purpose of this case is validation of the Flow360 solver
against experimental data and other CFD solvers, primarily CFL3D. In
this study, the simulations are performed using the SA and
:math:`k-\omega` SSTm turbulence models. The case is part of `the NASA Turbulence Modelling Website <https://turbmodels.larc.nasa.gov/backstep_val.html>`_ and has
also been evaluated by other CFD solvers such as `OVERFLOW in Jespersen et al <https://turbmodels.larc.nasa.gov/Papers/NAS_Technical_Report_NAS-2016-01.pdf>`_. The experimental data includes surface
pressure, skin friction data, velocity and turbulent shear stress
profiles upstream and downstream of the backward facing step `in Driver et al <https://arc.aiaa.org/doi/10.2514/3.8890>`_. The primary objective of this case is
validation for the prediction of reattachment point downstream of the
step. Additionally, verification can be performed by comparing with the
CFL3D code. Comparisons were also performed using freestream and
subsonic inflow boundary conditions.

Simulation setup
----------------

A single structured grid with four zones is used, available at the
`Backward facing step case page of the TMR website <https://turbmodels.larc.nasa.gov/backstep_grids.html>`_. The second finest grid (257x257,
97x257, 385x449, 129x449) is used to perform direct comparisons with
CFL3D on the same grid. The grid used in Flow360 was modified for the names and number of boundaries, and is available `here <https://simcloud-public-1.s3.amazonaws.com/verification/backstep/BackStep_Lvl4_Grid.cgns>`_. A detailed grid convergence study is not
performed here. The grid topology for a coarser level 2 grid is shown in :ref:`Fig1_BackStepMesh`.

.. _Fig1_BackStepMesh:
.. figure:: Figures/GridBackward.png
   :align: center
   :scale: 80%
  
   Grid topology for the backward facing step study, Level 2 shown (two levels finer grid used in current study).

The flow conditions were set to 36,000 Reynolds number and a Mach number of 0.128. A reference temperature of 298.333K was used. Two different boundary conditions were examined for the inflow boundary condition. First, a standard freestream boundary condition was used with a specified freestream Mach number. Secondly, a subsonic inflow boundary condition was also used, with a specified pressure and temperature ratio calculated based on isentropic flow relationships (total pressure ratio = 1.011515853 and total temperature ratio = 1.0032768, based on M = 0.128). The outflow boundary condition was set through a static pressure ratio iteratively to ensure that the velocity profile upstream of the backward facing step matched CFL3D data. An outlet static pressure ratio of 1.0108 was used for the freestream inlet boundary condition and 0.99025 for the subsonic inflow boundary condition. Fully-turbulent calculations were performed with the ratio of the freestream value of the SA turbulence field variable (relative to laminar) set to 3, set in Flow360 as a turbulent viscosity ratio of 0.210438. For the k- |omega| SSTm calculations a turbulent viscosity ratio of 0.009 was used. The case layout is shown in detail in :ref:`Fig2_BackStepCaseLayout`, whereas the solver inputs are summarized in :ref:`Tab1_BackStep_SolverInputs`.

.. _Fig2_BackStepCaseLayout:
.. figure:: Figures/CaseLayoutBackward.png
   :align: center
   :scale: 80%

   Summary of case boundary conditions and conditions for the backward facing step study.

.. _Tab1_BackStep_SolverInputs:
.. csv-table:: Solver inputs for the backward facing step study.
   :file: Tables/backstep_tab1_keyflowconfig.csv
   :widths: 30, 20
   :align: center
   :header-rows: 1
   :delim: @

The simulations were performed using a CFL number of 100 ramped up over the initial 2000 iterations. A reduction in CFL to a value of 5 was required at the end of the simulations to ensure convergence to steady state. Contrary to reported results `by Jespersen et al <https://turbmodels.larc.nasa.gov/Papers/NAS_Technical_Report_NAS-2016-01.pdf>`_ and on the `TMR website <https://turbmodels.larc.nasa.gov/backstep_val_sst.html>`_ steady-state convergence was obtained when using the k- |omega| SSTm model, with only a minor oscillation in the final loads. For the majority of cases, a steady residual drop of at least 1e-10 for the flow residuals and 1e-8 for the turbulence residuals was achieved. 


Numerical Results
-----------------

Firstly, the reattachment locations are compared between Flow360, experimental data and different CFD solvers shown in :ref:`Tab2_Reattach`. Note the OVERFLOW code uses the baseline k- |omega| SST model, whereas the k- |omega| SSTm model is used by CFL3D and Flow360.

.. _Tab2_Reattach:
.. csv-table:: Comparison of reattachment locations between different CFD solvers and experiment for the backward facing step using the SA and k- |omega|-SSTm turbulence models.
   :file: Tables/backstep_tab2_reattach.csv
   :widths: 15, 10, 15
   :align: center
   :header-rows: 1
   :delim: @

The reattachment location predictions show excellent agreement with CFL3D and OVERFLOW predictions. The impact of the inlet boundary condition is not significant, although it must be noted that a different static pressure ratio was used at the outlet, to ensure the boundary layer shape upstream of the step. The SA model predicts earlier reattachment compared to experiment, whereas the k- |omega| SSTm model predicts a delayed reattachment location. 

To verify that correct inflow profile is applied and to verify the outlet static pressure ratio value, the U velocity and turbulent shear stress profiles are compared upstream of the backward facing step at x/H=-4, shown in :ref:`Fig3_BackStepInflowProfiles`.

.. _Fig3_BackStepInflowProfiles:
.. figure:: Figures/Upstream_Profiles.png
   :align: center

   Comparison of velocity and turbulent shear stress profiles at x/H=-4 between experimental data, Flow360, and CFL3D for two turbulence models.

Very good agreement is obtained for the upstream velocity and turbulent shear stress profiles when comparing Flow360 with CFL3D data. The velocity profile also shows very good agreement with experimental data, with a slight overprediction of the peak turbulent shear stress magnitude. The surface pressure and skin friction coefficients are analysed next, and are compared with the CFL3D code and experimental data in :ref:`Fig4_BackStepCP1`-:ref:`Fig7_BackStepCF2`. The surface pressure curves were translated to ensure a surface pressure coefficient of zero at x/H=40, as suggested on the `TMR backward step case study page <https://turbmodels.larc.nasa.gov/backstep_val.html>`_.

.. _Fig4_BackStepCP1:
.. figure:: Figures/SurfacePressureSA.png
   :align: center
   
   Comparison of the surface pressure coefficient data between Flow360, CFL3D and Experiment for the SA turbulence model.

.. _Fig5_BackStepCP2:
.. figure:: Figures/SurfacePressureSST.png
   :align: center
   
   Comparison of the surface pressure coefficient data between Flow360, CFL3D and Experiment for the k- |omega| SSTm turbulence model.
.. _Fig6_BackStepCF1:
.. figure:: Figures/SkinFrictionSA.png
   :align: center

   Comparison of the surface skin friction coefficient data between Flow360, CFL3D and Experiment for the SA turbulence model.

.. _Fig7_BackStepCF2:
.. figure:: Figures/SkinFrictionSST.png
   :align: center

   Comparison of the surface skin friction coefficient data between Flow360, CFL3D and Experiment for the k- |omega| SSTm turbulence model.

Excellent agreement is obtained between the Flow360 and CFL3D results for both the SA and k- |omega| SSTm model. The main region where differences can be seen is at x/H = 0.0 where a spike can be seen in the skin friction CFL3D skin friction data not present in the Flow360 results, which in turn exhibit a spike in the surface pressure coefficient data. This discontinuity is likely to be due to the high :math:`y^+` present on the backward facing step surface. The different inlet boundary condition leads to a slightly lower magnitude of the negative pressure peak and lower skin friction for the subsonic inflow, although these differences are fairly insignificant. Compared to experiments, the use of the k- |omega| SSTm model improves the skin friction curve slope as the flow reattaches compared to the SA model. For the surface pressure coefficient, the agreement is improved in the region of the separated flow, although better agreement is seen with experiment with the SA model after the flow reattaches.

Further analysis is performed by comparing the velocity profiles including the turbulent shear stress at four locations downstream of the backward facing step with data from CFL3D and experiment. The data is normalized by a :math:`U_{ref}` for the U and V velocities, and :math:`U_{ref}^2` for the turbulent shear stress. The velocity profiles for the two turbulence models are compared in :ref:`Fig8_BackStep_VelProf`. The data was extracted using two line probes with uniformly spaced points in Tecplot (one finer line probe for the inner portion of the boundary layer and one coarser line probe for the outer portion). This was done, as the gradients were computed in Tecplot and using the direct CFD data leads to discontinuities in the turbulent shear stress profiles.

.. _Fig8_BackStep_VelProf:
.. figure:: Figures/Downstream_Profiles.png
   :align: center

   Comparison of velocity and turbulent shear stress profiles at four locations downstream of the backward facing step between experimental data, Flow360, and CFL3D for two turbulence models.

The velocity and turbulent shear stress profiles once again indicate that the use of the subsonic inflow boundary condition does not have a significant impact although some minor variation can be seen in the turbulent shear stress profile for the k- |omega| SSTm model at x/H=4. Comparing CFL3D and Flow360 results, the velocity profiles are nearly identical for both turbulence models. A slightly higher turbulent shear stress is seen between y/H=0 and y/H=1 for Flow360 when compared to CFL3D for the SA model, which is likely to be due to the high :math:`y+` on the backward facing step wall. For the k- |omega| SSTm some slight variation can only be seen at x/H=4. Compared to experiment, the SA model predicts better agreement at location x/H=6 and x/H=10 whereas the k- |omega| SSTm model performs better at x/H=1 and x/H=4, which is seen for both the velocity and turbulent shear stress profiles.

Overall, the results between Flow360, experimental data and other CFD codes show a high degree of consistency leading to further validation and verification of the Flow360 solver. The solver version release-22.2.3.0 was used throughout this study.
