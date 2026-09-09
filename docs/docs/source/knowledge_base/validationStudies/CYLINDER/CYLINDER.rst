.. _cylinder2dValidationStudy:

.. |agr|  unicode:: U+003B1 .. GREEK SMALL LETTER ALPHA
.. |omega|    unicode:: U+03C9 .. OMEGA SIGN


Scale-Resolving Simulations Past a Circular Cylinder
====================================================

Introduction
------------

The main purpose of this validation study is the verification and validation of the scale-resolving techniques implemented in the Flow360 solver to represent turbulence. A Detached-Eddy Simulation is a three-dimensional unsteady solution based on a single turbulence model that functions as a sub-grid-scale model in a fine-grid region for a large-eddy simulation (LES) and as a Reynolds-averaged Navier-Stokes (RANS) model where the grid is not fine. Here, we are comparing the grid spacing with the thickness of the turbulent layer.
It is possible for a DES grid to become excessively refined in some regions to capture a flow feature. In that case, the original DES length scale can become smaller than the boundary-layer thickness, which causes an early activation of the LES mode which is undesirable.
This issue is known as "modeled-stress depletion" (MSD). A general approach to address MSD is to detect and shield the attached boundary layers and delay the activation of the LES mode even on grids where the DES length scale, which is proportional to the grid spacing, becomes smaller than the boundary-layer thickness. This approach was proposed by Spalart et al in 2006 as Delayed Detached-Eddy Simulation (DDES) (pure DES was introduced in 1997).

Turbulent flow contains a broad range of scales of length and time. The largest scales are related to the geometry and boundary conditions, while at the smallest length scales energy is dissipated by molecular viscosity.
Simulations that capture all length scales of motion through a numerical solution of the Navier-Stokes (NS) equations are called direct numerical simulation (DNS) , which is computationally very expensive. The RANS approach solves equations averaged over time; however, RANS has been inaccurate when applied to flows with significant large-scale unsteadiness.
The Large eddy simulation (LES) approach is an intermediate in computational complexity to address incompatibility of RANS in dealing with unsteady physics.
It considers that transport of momentum and energy is mostly due to the unsteady features in the larger length scales being resolved in time and space and the effect of smaller length scales can be represented by using subgrid-scale (SGS) models. The implicit LES (ILES) relies on the mesh resolution and the stabilization of the algorithm, and doesn't have an explicit SGS model.

The Roe scheme is an approximate Reimann solver based on a Godunov type scheme that solves a localized Reimann problem to calculate the flux.
The low-dissipation Roe is a modification of the Roe scheme to consider low Mach number problems and to achieve lower numerical dissipation in the range of higher resolved wave numbers for the scale-resolving simulations.
Both the standard and the low-dissipation schemes are considered for this validation study.

In this validation study, DDES and ILES simulations are performed for the cross flow over an infinite span circular cylinder to validate smooth separation where it provides the opportunity to show its competitiveness with both RANS and LES. We considered simulations with laminar separation (LS) at a Reynolds number 20,000. A DDES simulation is attractive because of the known issue of "gray area" between RANS and LES regions. When a rapid new large-scale instability affects the turbulence, both pure LES and DDES are very reasonable, especially when the separation happens due to sharp edges. The cylinder is a more challenging and better test case for testing the "grey-area failures" as well as the accuracy in predicting smooth-body separation which depends on the Reynolds number, surface curvature, pressure gradients, and other aspects of the flow.


Simulation Setup
----------------

The circular cylinder test case is based on experimental tests at different Reynolds numbers at which the flow undergoes laminar and turbulent separations. Experimental results for cross flow over a circular cylinder can be found in
`Turbulence effect on crossflow around a circular cylinder at subcritical Reynolds numbers <https://ntrs.nasa.gov/citations/19830005116>`_ and `Fluctuating lift on a circular cylinder: review and new measurements <https://www.sciencedirect.com/science/article/pii/S0889974602000993>`_.
For numerical results, one can refer to `Detached-Eddy Simulations Past a Circular Cylinder <https://link.springer.com/article/10.1023/A:1009901401183>`_ and `Large-Eddy Simulation of the Flow Over a Circular Cylinder at Reynolds Number 2 × 10⁴ <https://link.springer.com/article/10.1007/s10494-013-9509-1>`_.
The freestream Mach number is equal to 0.03 with a Reynolds number of 20,000.

The boundary conditions are shown in :ref:`cylinder_bc`. In order to study the flow over a "2D" bluff-body, it is necessary to perform 3D simulations and consider the impact of the boundary condition in the third direction according to `Travin et. al <https://link.springer.com/article/10.1023/A:1009901401183>`_.

.. _cylinder_bc:
.. figure:: Figures/Cylinder_BoundaryConditions.png
   :align: center
   :width: 100%

   Summary of boundary conditions and flow conditions for the cross flow over the circular cylinder case.

We used periodic boundary conditions and a spanwise length of :math:`L_{z}` = 4 :math:`\times` D to resolve the flow structure in comparison to the experiments.
Because the freestream turbulence influences the instabilities in the separated shear layer, it is necessary to use periodicity. The entire domain has a dimension of 60D :math:`\times` 60D :math:`\times` 4D.

In general, we generated the grids with cubic hexahedron cells in the region away from the cylinder surface and sizing them according to the desired mesh resolution.
It is recommended to refine the grid in all directions simultaneously. For the grid refinement study, three grid levels are generated. The mesh node statistics for the three mesh levels are presented in :ref:`grid_levels` with the meshes shown in :ref:`cylinder_mesh`.

For this validation study, the grid refinement is a test for quality and sensitivity of solution accuracy to mesh resolution rather than a traditional convergence study. In the present work, we used 160 grid points in the coarse mesh, 225 points in the medium mesh and 320 in the fine mesh over the span length.
For the coarse level, 500 mesh points; for the medium level, 703 points; and for the fine level, 1001 points, are distributed uniformly over the cylinder's circumference. Number of mesh points in the x-y slice mesh outside the mesh boundary layer is shown in the table as "Wake".
The mesh resolving the boundary layer is built with hyperbolic extrusion and is thick enough to include the physical boundary layer.

.. _grid_levels:
.. csv-table:: Grid levels for mesh refinement study. 
   :file: Tables/cylinder_tab1_gridref.csv
   :widths: 12, 20, 20, 20, 14, 14
   :align: center
   :header-rows: 1

.. _cylinder_mesh:
.. figure:: Figures/Cylinder_Mesh.png
   :align: center
   :width: 100%

   Three grid levels generated for the cross flow over the cylinder.

For an unsteady simulation, the turbulent wake is considered as fully established after a duration of :math:`T \cdot U_{\infty}/D` = 350 by `Lysenko et. al <https://link.springer.com/article/10.1007/s10494-013-9509-1>`_, where T is the period, U is the freestream velocity and :math:`T =1/f`. The frequency of the vortex sheddings can be obtained by :math:`St=\frac{f \cdot D}{U_{\infty}}`.
The time step size is decreased in three phases in order to establish the unsteady solution after 35 vortex sheddings. Afterwards, the solution is restarted with a smaller fixed time step size for a further 20 vortex sheddings. Gradually decreasing the time step size helps to quickly develop the solution for the initial 35 vortex sheddings. Moreover, time-step refinement helps to reduce the sources of numerical errors.
In this case, the initial time step size of 0.54 is used. Then it is reduced to 0.35 and eventually to 0.14. All the time step sizes are non-dimensionalized using standard Flow360 conventions. For a quantitative validation of the present simulation, the solution is sampled over the last :math:`N_{vp}` = 20 shedding cycles.

If the ILES simulation with the low-dissipation scheme is unstable, a smaller time-step size with a fixed, large CFL values can be used. For this case, the CFL number is fixed at 100,000.
The low-dissipation factor can be adjusted to reduce the dissipation of the numerical scheme.

Grid Refinement Study
---------------------

Results of the grid refinement study are presented in :ref:`cylinder_refinement`. The averaging operator is shown by :math:`\langle \rangle`.
In this table, results on three grid levels for the time-averaged drag coefficient, root mean square of the lift coefficient :math:`\langle C_{l}\rangle`, Strouhal number :math:`\langle S_{t}\rangle`, and separation angle are presented.
The results for the fine grid match the results reported in literature for this Reynolds number accurately. Based on the experimental results for Re = :math:`2 \times 10^{4}`, the root mean square of the lift coefficient :math:`C^{\prime}_{l}` varies between 0.42 and 0.63.
The experimental and numerical results in literature also assess :math:`\langle C_{d}\rangle` = 1.2. The Strouhal number measured from the experiment is :math:`St_{exp}` = 0.19 and the separation angle is :math:`\theta_{sep}` = 78 |deg|. A summary of experimental and numerical results at this Reynolds number can be found in `Lysenko et. al <https://link.springer.com/article/10.1007/s10494-013-9509-1>`_.
For the simulations results presented in :ref:`cylinder_refinement`, the Strouhal number is determined from FFT of the :math:`C_{l}` time signal. The LDRoe scheme shows better agreement with experiment on medium mesh in comparison to the Roe scheme.

.. _cylinder_refinement:
.. csv-table:: Grid refinement study for ILES simulations with Roe and LDRoe schemes. 
   :file: Tables/cylinder_tab2_gridrefres.csv
   :widths: 20, 10, 20, 10, 10, 7, 7
   :align: center
   :header-rows: 1

The time-averaged pressure coefficient over the mid section is plotted in :ref:`cylinder_avg_cp_refinement` for three mesh levels.
For both ILES simulations with the Roe and LDRoe schemes, the pressure coefficient for the fine mesh matches the experimental results reported by Norberg with a high degree of accuracy. 
However, with the Roe numerical scheme, the predictions using the coarse and medium meshes, follow the trends for the averaged pressure coefficient but they both are inaccurate after the separation point along the downstream face of the cylinder.
On the contrary, with the low-dissipation numerical scheme, the results for the coarse and medium meshes are very close to the results on the fine mesh. 
This indicates that with the low-dissipation scheme on coarser mesh levels, more accurate results are achievable and the scheme is less dependent on mesh resolution or degrees of freedom.

.. _cylinder_avg_cp_refinement:
.. figure:: Figures/Cylinder_MeshRef.png
   :align: center
   :width: 100%

   Pressure coefficient comparison for three grid levels. Level 3 fine mesh, 43M nodes (red). Level 2 medium mesh, 24M nodes (green). Level 1 coarse mesh, 7.5M nodes (blue). 

The time-averaged drag (:math:`C_{d}`) and root-mean-square lift (:math:`C_{l}`) coefficients are shown in :ref:`cylinder_gci_refinement` against grid convergence index (GCI). For both numerical schemes, the ILES simulations with the low-dissipation scheme are converging
faster in comparison with the Roe scheme with increasing mesh resolution and showing asymptotic behavior.

.. _cylinder_gci_refinement:
.. figure:: Figures/Cylinder_gridConv.png
   :align: center
   :width: 100%

   Grid convergence figure for the time-averaged drag (:math:`C_{d}`) and root-mean-square lift (:math:`C_{l}`) coefficients on three grid levels for the cross flow over the cylinder.

Time-Dependent Flow Field Results
---------------------------------

First, the loading time-histories are examined to investigate the impact of the numerical scheme on the load oscillation magnitudes and shedding cycle behavior.
The time histories of aerodynamic coefficients for different methods are compared in :ref:`cylinder_time_coefficients`.
It shows the time histories of the lift (:math:`C_{l}`) and drag (:math:`C_{d}`) coefficients for a period of :math:`\approx` 20 vortex shedding cycles for four simulations on the fine mesh.
In all figures, an initial transient of length 35 was removed. The figures show that removing of 10 more cycles for DDES-SA simulations and for simulations with the low-dissipation scheme would have been desirable.
All sub-figures show strong modulations of the shedding phenomenon. A higher drag is found with larger lift oscillation amplitude. The following drag coefficient comparison relate to this flow phenomenon.
Compared to simulations with Roe scheme, low-dissipation schemes shows lower lift amplitude as shedding cycles are established.

.. _cylinder_time_coefficients:
.. figure:: Figures/Cylinder_timeHistory.png
   :align: center
   :width: 100%

   Time history of the lift :math:`C_{l}` and drag :math:`C_{d}` coefficients (:math:`\approx` 20 vortex shedding cycles) obtained from ILES and DDES simulations.

:ref:`cylinder_time_method` shows the time histories of the aerodynamic coefficients for a the same period of :math:`\approx` 20 vortex shedding cycles for the ILES and URANS simulations with the Roe scheme.
Similar to previous simulations, an initial transient of length 35 shedding cycles is removed. The modulations of the shedding phenomenon is present in both. However, the Unsteady RANS simulation doesn't show any change over the coarse of time in comparison to the ILES simulation. 

.. _cylinder_time_method:
.. figure:: Figures/Cylinder_uransvsILES.png
   :align: center
   :width: 100%

   Time history of the lift :math:`C_{l}` and drag :math:`C_{d}` coefficients (:math:`\approx` 20 vortex shedding cycles) obtained from ILES and URANS simulations.

The iso-surface of Q-criterion is shown in :ref:`cylinder_qCriterion` to clearly demonstrate the three dimensionality of the solutions. 
The iso-surface figures for all simulations shows the vortex core that starts from the shear layer on the upper and lower surface of the cylinder.
The ILES simulations are capturing the much small vortices near the cylinder after the separation. Both simulations with low-dissipation schemes either ILES or DDES shows finer vortex features.

.. _cylinder_qCriterion:
.. figure:: Figures/Cylinder_qCriterion.png
   :align: center
   :width: 100%

   Iso-surfaces obtained from ILES and DDES simulations for the flow over a circular cylinder at Re= :math:`2 \times 10^{4}`

From the side view, the iso-surface of Q-criterion is shown in :ref:`cylinder_qCriterion_vortex` that reveals a dominant 2D von Karman vortex-shedding mode in all simulations.
The ILES and DDES simulations with the Roe scheme show very similar vortex features in the wake. The ILES simulation is slightly better in capturing flow features.
The ILES and DDES simulations with the low-dissipation scheme captures finer flow features in the wake and near the cylinder after the separation.
The ILES simulation with the low-dissipation scheme shows finer vortex structures in comparison to others.

.. _cylinder_qCriterion_vortex:
.. figure:: Figures/Cylinder_qCriterion_vortex.png
   :align: center
   :width: 100%

   Iso-surfaces obtained from ILES and DDES simulations for the flow over a circular cylinder at Re= :math:`2 \times 10^{4}`

In order to make a comparison between the different methods, the iso-surface of Q-criterion for an unsteady RANS, DDES based on the SA turbulence model, ILES with the Roe scheme and ILES with the low-dissipation Roe scheme is shown in :ref:`cylinder_qCriterion_method`.
This comparison shows that by adding to the fidelity of an unsteady simulation, more complete flow features can be captured. The DDES versus ILES with the Roe scheme primarily differ in the region of boundary layer separation. The ILES with the low-dissipation Roe scheme resolves the downstream vortex structures significantly better than other methods, improving the accuracy and reliability.

.. _cylinder_qCriterion_method:
.. figure:: Figures/Cylinder_URANSqCriterion.png
   :align: center
   :width: 100%

   Iso-surfaces obtained from unsteady RANS, DDES, ILES with the Roe and low-dissipation Roe simulations for the flow over a circular cylinder at Re= :math:`2 \times 10^{4}`

The instantaneous contours of vorticity magnitude on the x-y plane for the ILES and DDES simulations are shown in :ref:`cylinder_vorticity_magnitude`. All instantaneous frames are at :math:`T \cdot U_{\infty}/D` = 550.
They all are on the fine mesh. The Roe scheme does not capture the small flow features in the recirculation region, while flow features are captured very well in this region and near the laminar boundary layer separation with the low-dissipation scheme.

.. _cylinder_vorticity_magnitude:
.. figure:: Figures/Cylinder_vorMag.png
   :align: center
   :width: 100%

   Vorticity magnitude contours in the x-y plane obtained from ILES and DDES simulations for the flow over a circular cylinder at Re= :math:`2 \times 10^{4}`

Time-Averaged Flow Field Results
--------------------------------

In this section, time-averaged flow field results are compared against experiment and numerical results. It shows how well different methods predict time-averaged flow quantities for the cross flow over the circular cylinder with unsteady flow behavior. :ref:`cylinder_time_coefficients` shows the time-averaged streamlines in the x-y plane for the ILES and DDES simulations. Besides the main recirculation bubble, vortices attached to the downstream surface of the cylinder are present in all the results.
The existence of the small secondary vortices at the back side of the cylinder is confirmed by experimental results by `Lysenko et. al <https://link.springer.com/article/10.1007/s10494-013-9509-1>`_. This shows that several separation angles beside the primary separation exist.
The primary separation based on the experimental results is at :math:`\langle \theta_{s1} \rangle=` 78 |deg| and the secondary separation is at :math:`\langle \theta_{s2} \rangle=` 108 |deg|.
The DDES simulations show a greater recirculation length in the wake region. The recirculation length :math:`\langle L_{r} \rangle` indicates the distance between the base of the cylinder and the sign change of the centerline mean stream-wise velocity.
Unfortunately, there is no experimental results for the recirculation length at this Reynolds number.

.. _cylinder_streamline_avg:
.. figure:: Figures/Cylinder_streamline.png
   :align: center
   :width: 100%

   Time-averaged streamline in the x-y plane obtained from ILES and DDES simulations for the flow over a circular cylinder at Re= :math:`2 \times 10^{4}`

The time-averaged pressure coefficient over the cylinder slice in the x-y plane is shown in :ref:`cylinder_Cp_avg`. 
On the left, results for the ILES simulation are shown and match well with the experimental results provided by Norberg at Re= :math:`2 \times 10^{4}`. The ILES simulation with the low-dissipation scheme leads to slightly better agreement between :math:`\langle \theta \rangle=` 78 |deg| and :math:`\langle \theta \rangle=` 100 |deg|.
On the right, results for the DDES simulations are presented. The result for the DDES simulation with the Roe scheme is closer than the LDRoe predictions to the experiment in the backward side of the cylinder.
The disagreement between the ILES and DDES simulations for the time-averaged pressure coefficients indicate the impact of laminar separation at this Reynolds number.
For the time-averaged pressure coefficient, better correlation is expected when turbulent separation is considered.

The mean base suction coefficient :math:`\langle C_{p,b} \rangle` is determined as the mean pressure coefficient on the cylinder's surface at :math:`\langle \theta_{s1} \rangle=` 180 |deg|.
It is strongly related to the mean drag coefficient. The calculated value of the mean base suction coefficient by ILES with the Roe scheme is :math:`\langle C_{p,b} \rangle=` -1.16 and by ILES with LDRoe scheme is :math:`\langle C_{p,b} \rangle=` -1.2. Both have good agreement with the experiment.
The experimental value reported by Norberg is :math:`\langle C_{p,b} \rangle=` -1.197. The calculated value by DDES with the Roe scheme is :math:`\langle C_{p,b} \rangle=` -1.06 and by DDES with LDRoe scheme is :math:`\langle C_{p,b} \rangle=` -0.962, which doesn't show good agreement with the experiment in comparison to ILES simulations.

.. _cylinder_Cp_avg:
.. figure:: Figures/Cylinder_Cp.png
   :align: center
   :width: 100%

   Time-averaged pressure coefficient :math:`C_{p}` obtained from ILES and DDES simulations for the flow over a circular cylinder at Re= :math:`2 \times 10^{4}`

To compare the flow separation, the time-averaged skin-friction coefficients are shown in :ref:`cylinder_Cf_avg`. Both ILES and DDES simulations with the Roe and LDRoe schemes have good agreement. The primary separation angle is :math:`\langle \theta \rangle=` 78 |deg| based on the experimental results provided by Norberg.
The separation angle for the ILES simulation with the Roe scheme is 81 |deg| and with the LDRoe scheme is 82 |deg| that both are in agreement with the experimental result.
For the DDES simulations, the separation angles are also similar. This indicates both ILES and DDES methods and both Roe schemes on the fine mesh are closely predicting the expected primary separation angle.

.. _cylinder_Cf_avg:
.. figure:: Figures/Cylinder_Cfx.png
   :align: center
   :width: 100%

   Time-averaged skin-friction coefficient :math:`C_{f}` obtained from ILES and DDES simulations for the flow over a circular cylinder at Re= :math:`2 \times 10^{4}`

For a RANS, unsteady RANS, DDES and ILES simulations, time-averaged pressure :math:`C_{p}` and skin-friction :math:`C_{f}` coefficients are compared in :ref:`cylinder_CpCf_method` to make a comparison between different CFD methods for the cross flow over the circular cylinder.
The steady RANS is very far from the experiment because of unsteady flow behavior. The unsteady RANS fails to predict the flow near and after the separation because of not accurately capturing the instabilities in the separated shear layer. The DDES simulation is close to experiment and the ILES simulation matches almost exactly with the experiment.
Meanwhile, time-averaged quantities for these simulations are compared in :ref:`cylinder_comparison_method`.

.. _cylinder_CpCf_method:
.. figure:: Figures/Cylinder_CpURANS.png
   :align: center
   :width: 100%

   Time-averaged pressure coefficient :math:`C_{p}` and skin-friction coefficient :math:`C_{f}` obtained from RANS, URANS, DDES and ILES simulations for the flow over a circular cylinder at Re= :math:`2 \times 10^{4}`

.. _cylinder_comparison_method:
.. csv-table:: Wall-averaged quantities for the cross flow simulations over the circular cylinder with different methods on the fine mesh. 
   :file: Tables/cylinder_tab3_method.csv
   :widths: 10, 20, 15, 10, 10, 10, 10
   :align: center
   :header-rows: 1

The mean stream-wise velocity :math:`\langle u \rangle` along the centerline is shown in :ref:`cylinder_mean_velocity`. The minimum values of the mean stream-wise velocity predicted are :math:`\langle u_{min} \rangle/U_{\infty}=` -0.27 for the ILES simulation with the low-dissipation scheme and :math:`\langle u_{min} \rangle/U_{\infty}=` -0.24 for the DDES 
simulation with the low-dissipation scheme. For the ILES and DDES simulations with the Roe scheme, the :math:`\langle u_{min} \rangle/U_{\infty}` respectively is -0.32 and -0.31.

.. _cylinder_mean_velocity:
.. figure:: Figures/Cylinder_meanVelo.png
   :align: center
   :width: 100%

   Mean stream-wise velocity :math:`\langle u \rangle/U_{\infty}` obtained from ILES and DDES simulations for the flow over a circular cylinder at Re= :math:`2 \times 10^{4}`

An overview of the simulations and flow features from available experimental and numerical results are provided in :ref:`cylinder_avg_statistics`. This table shows a comparison between the values obtained from the experiment and values obtained from different numerical methods.
The results for the ILES simulations with both Roe and LDRoe schemes agree well with the experimental results.
The ILES simulation with low-dissipation scheme performs better in predicting flow feature in the recirculation wake. 

.. _cylinder_avg_statistics:
.. csv-table:: Overview of the experimental, ILES and DDES works for the flow over a circular cylinder at Re= :math:`2 \times 10^{4}`. 
   :file: Tables/cylinder_tab4_statistics.csv
   :widths: 10,20,8,4,2,2,8,8,5,8,10,8,4
   :align: center
   :header-rows: 1

| Exp = Experiment
| LES = Large Eddy Simulation
| NSGS = No Subgrid-Scale
| TKE = Turbulence Kinetic Energy
| I = Incompressible
| DDES = Delayed Detached-Eddy Simulation
| ILES = Implicit Large Eddy Simulation

| :sup:`(1)` Schewe G. On the force fluctuations acting on a circular cylinder in crossflow from subcritical up to transcritical Reynolds numbers. Journal of fluid mechanics. 1983 Aug;133:265-85.
| :sup:`(2)` Yokuda S, Ramaprian BR. The dynamics of flow around a cylinder at subcritical Reynolds numbers. Physics of Fluids A: Fluid Dynamics. 1990 May;2(5):784-91.
| :sup:`(3)` Norberg C. Fluctuating lift on a circular cylinder: review and new measurements. Journal of Fluids and Structures. 2003 Jan 1;17(1):57-96.
| :sup:`(4)` Lim HC, Lee SJ. Flow control of circular cylinders with longitudinal grooved surfaces. AIAA journal. 2002 Oct;40(10):2027-36.
| :sup:`(5)` Salvatici E, Salvetti MV. Large eddy simulations of the flow around a circular cylinder: effects of grid resolution and subgrid scale modeling. Wind and Structures. 2003 Nov 1;6(6):419-36.
| :sup:`(6)` Wornom S, Ouvrard H, Salvetti MV, Koobus B, Dervieux A. Variational multiscale large-eddy simulations of the flow past a circular cylinder: Reynolds number effects. Computers & Fluids. 2011 Aug 1;47(1):44-50.
| :sup:`(7)` Lysenko DA, Ertesvåg IS, Rian KE. Large-eddy simulation of the flow over a circular cylinder at Reynolds number 2×10 :sup:`4`. Flow, turbulence and combustion. 2014 Mar;92:673-98.

Conclusions
-----------

This validation study suggests the following recommendations for LES and DDES simulations:

- For 2D bluff-body flows, 3D simulations are necessary. The spanwise length is recommended to be at least :math:`3 \times D` to resolve the flow structures observed in the experiment.

- For the modest subcritical regime, freestream turbulence influences the instabilities in the separated shear layer. For this reason, using periodicity is necessary and slipWall and symmetry should not be used.

- For simulations with a very low Mach number, it is necessary to have large enough domain to reduce the impact on the converged solution. Although larger domains need more physical steps for convergence.

- For bluff-body laminar separation flow:

  - Having enough mesh resolution in downstream wake is necessary for accurate results.

  - ILES simulations with the low-dissipation scheme provides the most accurate results compared to the experiment. In general, the low-dissipation scheme achieves more accurate results with less mesh resolution which indicates it is less dependent on the spatial degrees of freedom.

  - DDES simulations with the low-dissipation scheme are acceptable but are less accurate than ILES in the downstream side of the bluff-body. It is expected that for turbulent separation cases, DDES simulations will agree more closely with ILES simulations.

- For ILES simulations with the low-dissipation scheme, the simulation could be unstable, in that case a smaller time-step size with a fixed, large CFL values could help remedy this issue. Moreover, increasing the numerical low-dissipation factor helps to stabilize convergence. In this validation study, a low-dissipation factor of 0.5 is used.

- The Flow360 solver is validated against experimental data and other scale-resolving simulations for laminar separation flow over a circular cylinder. Good agreement is found for time-dependent and time-averaged flow features and quantitative results.
