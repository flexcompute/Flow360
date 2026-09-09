.. _acousticsValidationStudy:
.. |agr|  unicode:: U+003B1 .. GREEK SMALL LETTER ALPHA
.. |omega|    unicode:: U+03C9 .. OMEGA SIGN

.. _drag crisis: https://en.wikipedia.org/wiki/Drag_crisis

Aeroacoustics and Noise Simulation
==================================

Introduction
------------

The main purpose of this validation study is the verification and validation of the aeroacoustics and noise simulations with the Flow360 solver. 

The flow around an aerodynamic body is governed by the Navier-Stokes equations, which express the conservation of mass, momentum, and energy. Direct Numerical Simulation (DNS) serves as a method for directly solving these equations.
Achieving a comprehensive understanding of turbulent flow through DNS simulations demands a finely detailed grid and a small time step. The grid resolution must be fine enough to accurately capture the Kolmogorov length scale.
In cases of isotropic turbulence, the grid requirements for a DNS simulation intensify in proportion to the Reynolds number. Furthermore, DNS simulations encounter challenges beyond computational intensity. The inherent complexity of turbulent flows, characterized by a broad range of spatial and temporal scales, necessitates the capture of intricate details. 
This requirement for high resolution in both space and time amplifies the computational burden. Additionally, DNS simulations require careful consideration of boundary conditions, which can be challenging to accurately model in realistic aerodynamic scenarios. 
These challenges collectively contribute to the current impracticality of employing DNS simulations in aeronautical applications.

In applications involving high Reynolds numbers, a widely adopted methodology involves categorizing turbulent flow characteristics into mean and fluctuating components, resulting in the derivation of the unsteady Reynolds-averaged Navier-Stokes (URANS) equations. 
This practical approach facilitates the computational representation of turbulent phenomena. The significance of turbulence modeling in this context is crucial. It contributes indispensably by providing necessary closure through the approximation of the Reynolds stress tensor within the URANS equations. 
Among the array of turbulence models, the Spalart-Allmaras (SA) model stands out for its widespread adoption in external aerodynamics, because it finds a good balance between being accurate and not taking up too much computer power.
Consequently, the URANS framework, coupled with turbulence modeling, emerges as a valuable tool for the comprehensive simulation and comprehension of turbulent flows, particularly in practical engineering scenarios involving by high Reynolds numbers.

Studies have demonstrated that URANS lacks the capability to accurately predict unsteady flow behavior, particularly in cases of significant flow separation. Furthermore, its deficiency in scale-resolving capability limits its applicability for predicting broadband noise. 
Nonetheless, URANS can still be effectively utilized for design exploration and the prediction of tonal noise. Large-Eddy Simulation (LES) is commonly utilized for distinguishing larger turbulent motions from smaller scales. This is achieved by using a filter to partially resolve turbulence above a specific length scale. 
For scales below this threshold, turbulence is typically modeled empirically. When filtering is implemented, the filtered equations assume a similar form to the Unsteady Reynolds-Averaged Navier-Stokes (URANS) equations. 
The sub-grid stress tensor becomes crucial in LES, as it characterizes the unresolved scales and necessitates a modeling approach.
Filtering can be explicitly applied by using a Sub-Grid-Scale (SGS) model to represent the unresolved sub-grid scale, or implicitly through the utilization of numerical dissipation in discretization as a low-pass filter. In simpler terms, implicit filtering acts as a filter with a local grid spacing. 
When considering implicit Large-Eddy Simulation (LES), no explicit SGS model is incorporated. LES is a commonly employed method for aeroacoustic simulation, as it effectively resolves a broad range of large turbulent eddies. 
However, it's important to note that the grid resolution requirements for LES simulations of high Reynolds number flows scale proportionally with the Reynolds number and are considered computationally expensive.
More information about the influence of Reynolds number on drag coefficient can be found in `drag crisis`_.

In this validation study, the URANS, ILES, and DDES approaches are applied to different applications, and the outcomes are compared against experimental results.
The following sections are presented in this documentation:

1. :ref:`First Part: Cylinder Case <acousticsCylinder>`:

   - Comparison of pressure fluctuations
   - CFD-FWH vs. Direct-CFD
   - Verification using URANS approach

2. :ref:`Second Part: Rod-Airfoil Configuration <acousticsRod>`:

   - Validation of acoustic solver against experimental data 
   - Utilizing ILES approach

3. :ref:`Third Part: Two-Bladed Propeller Test Case <acousticsProp>`:

   - Further validation of acoustic solver against experimental data
   - Incorporating ILES and DDES approaches


.. _acousticsCylinder:

Cross Flow over Circular Cylinder
---------------------------------

In this instance, we validate the accuracy of the flow and acoustic solver (CFD-FWH). The freestream Mach number is set at 0.25, accompanied by a Reynolds number of :math:`5 \times 10^{6}`.

:ref:`Fig1_Cylinder_BC` illustrates the boundary conditions, mesh resolution, and microphone locations.

.. _Fig1_Cylinder_BC:
.. figure:: Figures/cylinder_bc.png
   :align: center
   :width: 100%

   Summary of boundary conditions, mesh details, and microphone locations for the cross-flow over the circular cylinder case.

We used periodic boundary conditions with a spanwise length of :math:`L_{z}` = 4 :math:`\times` D.
The mesh node statistics are detailed in :ref:`Tab1_Grid_Cylinder`.

.. _Tab1_Grid_Cylinder:
.. csv-table:: Grid properties for cross flow over the cylinder. 
   :file: Tables/cylinder_grid_tab1.csv
   :widths: 10, 20, 20, 30, 20
   :align: center
   :header-rows: 1
   :width: 70%

The purpose of this test case is to validate the acoustics results by comparing pressure fluctuations obtained from the acoustic solver with those directly extracted from the CFD data. 
In simulating the unsteady flow, URANS with the SA turbulence models is employed, using a second-order dual time-stepping method with a time step size of :math:`\frac{0.2 D}{c_{\infty}}`, corresponding to 100 time steps per vortex-shedding cycle.

Flow360's acoustic solver computes pressure fluctuations at six microphone locations. Specifically, three microphones are radially positioned at a constant polar angle, while three are placed circumferentially at an equal distance from the cylinder.
In :ref:`Fig1_Cylinder_BC`, microphone locations are highlighted with red dots. Detailed positions can be found in :ref:`Tab2_Mics_Positions`. 
The objective is to verify and validate the results of the acoustic solver against direct CFD data extraction.

The FWH equation is applied using only the “dipole” source terms on the cylinder surface (the Curle approximation at low Mach numbers).
The FWH equation was derived for 3D bodies finite in all directions, which this body is not. 
To reflect this, the pressure fluctuations in the acoustic region are calculated by adding the integral calculated on the primary cylinder of length 4D and the integrals calculated on a number of periodic “images” in both directions spanwise. 
As the image goes to large distances, its contribution dwindles and the series converges. This is crucial to the comparison with the (periodic) simulation field itself. 
In particular, for a finite body the pressure signal (:math:`p^{\prime}`) decays like 1/r for large r, but for 2D and periodic situations it decays only as fast as 1/sqrt(r). 
This was verified for microphones 1, 5, 2, and 3.

.. _Tab2_Mics_Positions:
.. csv-table:: Microphone positions around the cylinder. 
   :file: Tables/cylinder_mic_tab2.csv
   :widths: 30, 20, 20
   :align: center
   :header-rows: 1
   :width: 40%

The comparison of pressure fluctuations at the three microphones at the same radial distance from the cylinder is illustrated in :ref:`Fig2_Circum_PP`. 

.. _`Fig2_Circum_PP`:
.. figure:: Figures/cylinder_circum_now.png
   :align: center
   :width: 100%

   Pressure fluctuations for microphones at the same radius for the cross flow over the circular cylinder.

The frequency matching for pressure fluctuations from both the CFD solver and acoustic solver is excellent for all three radial observers. 
However, a discrepancy in amplitude is observed at microphone 4, tentatively attributed to the increased impact of 3D turbulent flow downstream of the body, where this observer is positioned.
The Quadrupole source terms which were ignored are strong in that region.

.. _Fig3_Radial_PP:
.. figure:: Figures/cylinder_radial_now.png
   :align: center
   :width: 100%

   Pressure fluctuations for microphones at the same angle for the cross flow over the circular cylinder 

Meanwhile, the pressure fluctuations at three microphones at the same angle are illustrated in :ref:`Fig3_Radial_PP`.
Frequency alignment is observed for all three radial microphones, with a slight mismatch in amplitude at microphone 3 due to mesh coarsening. 
Furthermore, a :math:`p^{\prime}` scaling proportional to :math:`1/\sqrt{r}` is achieved.
The internal consistency test is successful.

This comparison underscores the equivalence of pressure fluctuations from the acoustic solver to those directly computed from CFD calculations. 
However, emphasizing the importance of mesh resolution in the near-field region, the accuracy of CFD-FWH results is contingent upon the quality of the unsteady CFD calculation.

.. _acousticsRod:

Rod-Airfoil Configuration
-------------------------

The rod-airfoil configuration serves as a standard benchmark for investigating the interaction noise between different aircraft components. 
`Jacob et al. <https://doi.org/10.1007/s00162-004-0108-6>`_ :sup:`1` conducted the experiment in the wind tunnel at Ecole Centrale de Lyon (ECL). The primary goal of these experimental tests was to establish a comprehensive database against which CFD/CAA methods could be benchmarked.
In the study conducted by Jacob et al., they simulated basic geometric elements, resulting in two distinct types of noise: quasi-tonal noise due to the periodic shedding of vortices at the rod and broadband noise caused by the turbulent wake impinging on the airfoil. 
This setup provides a means to assess the CFD code's accuracy in predicting the shedding frequency of vortices from a bluff body and modeling the decay of turbulent structures in the wake. 
Widely employed for validating numerical methods in airframe noise prediction, this configuration is a prevalent reference in the field.

As per Jacob et al., the rod generates a von Kármán vortex street with a Strouhal number (:math:`St`) of :math:`St=0.19`. 
The rod's boundary layer maintains laminar characteristics, transitioning to turbulence in the wake. Instability waves emerge in the shear layers, engaging with two-dimensional structures. 
The ensuing turbulent wake interacts with the airfoil's leading edge, positioned one chord downstream.

Impinging on the airfoil, the large vortices split and form two smaller eddies, passing above and below the airfoil. 
Smaller vortices undergo distortion and don't make direct contact. Consequently, both direct and non-direct interactions between vortices and the airfoil can be examined. 
These interactions induce unsteady loads on the airfoil and alter the acoustic spectrum. Comparing the rod-airfoil configuration to the rod-only setup, the sound pressure level's tonal peak is higher in the former. 
The smaller vortices, significantly smaller than the airfoil's maximum thickness, introduce a broadband component to the spectrum. 
Furthermore, the tonal peak broadens due to the development of turbulence and the nonlinear effects of large vortices on the airfoil's leading edge.

Various researchers have undertaken comparisons among URANS, LES, and DES methodologies for this specific test case.
`Casalino et al. <https://doi.org/10.2514/2.1959>`_ :sup:`2` investigated unsteady compressible RANS simulations, while `Peth et al. <https://repository.tudelft.nl/islandora/object/uuid:b783ed06-3556-4cb1-8367-8c4aa4f26b15/datastream/OBJ>`_ :sup:`3` utilized a compressible implicit LES solver for their numerical study. 
`Galdeano et al. <https://doi.org/10.2514/6.2010-3702>`_ :sup:`4` conducted a study employing a compressible DES based on the SA turbulence model.
According to `Giret et al. <https://doi.org/10.2514/6.2012-2058>`_ :sup:`5`, given that the boundary layer's transition to turbulence and its detachment significantly impact shedding frequency, unsteady loads, and consequently the acoustic pressure spectrum, the effectiveness of the DES method may hinge on the model used to resolve boundary layers around the bluff-body.
They investigated the specified configuration employing an unstructured compressible Large Eddy Simulation (LES) code. Their study involved a comparative analysis of the influence of spanwise dimension and the sensitivity of rod/airfoil alignment on the prediction of noise.

In our validation study, we chose compressible implicit LES simulations to predict the acoustic pressure spectrum. 
The simulations are carried out using the Flow360 code, which is an unstructured compressible solver. Both Roe and low-dissipation Roe are used in this validation study.
Our study's primary aim is to evaluate the effectiveness of Flow360, coupled with the Ffowcs-Williams and Hawkings (FWH) code, in calculating far-field noise. 
Given our focus on evaluating far-field noise, using direct-CFD approach, similar to the cylinder test case in the previous section, becomes unfeasible. This is attributed to the necessity for a refined mesh across a considerable domain capable of propagating acoustic waves.
We intend to compare our findings with experimental data and other numerical results available in the existing literature.

Simulation Setup
^^^^^^^^^^^^^^^^

In this specific test case, the rod-airfoil configuration consists of a NACA0012 airfoil with a chord (C) of :math:`C=1 m` and a cylindrical rod with a diameter (d) of :math:`d=0.1C` that positioned one chord upstream.
The flow conditions encompass a uniform airflow with :math:`U_{in}=72 m/s`, :math:`T_{in}=293 K`, and :math:`\rho_{in}=1.2 kg/m^{3}`.
The Reynolds numbers based on the rod diameter (:math:`Re_{d}`) and the airfoil chord (:math:`Re_{C}`) are :math:`Re_{d}=4.8\times10^{4}` and :math:`Re_{C}=4.8\times10^{5}` respectively. The spanwise length is :math:`5d`.

:ref:`Fig4_rodAirfoil_BC` illustrates the boundary conditions, mesh resolution, and microphone locations.

.. _Fig4_rodAirfoil_BC:
.. figure:: Figures/rodAirfoil_bc.png
   :align: center
   :width: 100%

   Summary of boundary conditions, mesh and microphone locations for the rod and airfoil test case.

We implemented periodic boundary conditions with a spanwise length (:math:`L_{z}`) of :math:`L_{z}` = 5 :math:`\times` d.
Detailed mesh node statistics can be found in :ref:`Tab3_Grid_rodAirfoil`. 
Although a mesh refinement study around the rod was conducted, it is not presented in this particular validation study.

.. _Tab3_Grid_rodAirfoil:
.. csv-table:: Grid points for the rod and airfoil test case.
   :file: Tables/rodAirfoil_grid_tab3.csv
   :widths: 10, 10, 10, 15, 15, 20, 20
   :align: center
   :header-rows: 1
   :width: 70%

Aerodynamic Validation
^^^^^^^^^^^^^^^^^^^^^^
The iso-surface of Q-criterion plot shown in :ref:`Fig6_rodAirfoil_Q` reveals the vortex-shedding structure impinging on the airfoil.
Displayed in :ref:`Fig5_rodAirfoil_Cp` is the time-averaged pressure coefficient over the rod. A comparison is made between Implicit Large Eddy Simulation (ILES) with and without a low-dissipation scheme, and the experimental results provided by `Travin et. al <https://link.springer.com/article/10.1023/A:1009901401183>`_ :sup:`6`.
In both simulations, a notable accord is observed when evaluating against the experimental data. Particularly noteworthy is the ILES with a low-dissipation scheme, which demonstrates a marginally superior agreement beyond :math:`\theta` = 60 |deg|.

.. _Fig5_rodAirfoil_Cp:
.. figure:: Figures/rod_cp.png
   :align: center
   :width: 70%

   Time-averaged pressure coefficient :math:`C_{p}` obtained from ILES simulations with Roe and LDRoe schemes around the rod at Re= :math:`4.8 \times 10^{4}`.

.. _Fig6_rodAirfoil_Q:
.. figure:: Figures/rodAirfoil_qcomparison.png
   :align: center
   :width: 100%

   Iso-surfaces obtained from ILES simulations with and without the low-dissipation scheme at Re= :math:`4.8 \times 10^{4}`.

.. _Fig6_rodAirfoil_velocity:
.. figure:: Figures/rodAirfoil_velocity.png
   :align: center
   :width: 100%

   Velocity profiles comparison against experiment at three axial locations obtained from ILES simulation with LDRoe scheme for the rod and airfoil test case.

An assessment is conducted at three axial locations, as depicted in :ref:`Fig6_rodAirfoil_velocity`, to facilitate a comparative analysis of velocity profiles against experimental data.
The corresponding velocity profile comparisons are presented within the same figure. A fair agreement in mean velocities is discerned when evaluating against the experimental data.

Noise Evaluation
^^^^^^^^^^^^^^^^

The acoustic power spectral density (PSD) for angular positions 60 |deg|, 90 |deg|, and 120 |deg| is presented in :ref:`Fig8_rodAirfoil_spectra`. 
The numerical results exhibit a commendable match with both experimental data and other numerical results available in the literature across all microphone angular positions.
Similar to the cylinder case in the first section, the pressure fluctuations in the acoustic region are calculated by considering a number of periodic “images” in both directions spanwise.

.. _Fig8_rodAirfoil_spectra:
.. figure:: Figures/rodAirfoil_spectra.png
   :align: center
   :width: 80%

   Acoustic PSD comparison against experiment at different microphone positions obtained from ILES simulation with LDRoe scheme for the rod and airfoil test case.

The Strouhal peak is effectively captured for the two source surfaces. The computed Strouhal number based on the Sound Pressure Level (SPL) spectra for three observers is :math:`St=0.186`, which is in good agreement with the experimental value of :math:`St=0.19`. However, a more noticeable discrepancy at higher frequencies is observed, particularly at the downstream microphone (:math:`\theta` =60 |deg|).
This test case underscores the reliability of acoustic simulation using Flow360 with an Ffowcs-Williams and Hawkings (FWH) acoustic analogy for predicting noise in a rod-airfoil configuration.
The compressible Implicit Large Eddy Simulation (ILES) with a low-dissipation scheme on an unstructured mesh allows for high-fidelity resolution of acoustic sources and their propagation.
The FWH analogy efficiently extrapolates the pressure field in the far-field region at a relatively small computational cost. Meanwhile, the CFD-FWH method proves effective in predicting both the tonal and broadband components of the sound generated by the rod-airfoil configuration.

.. _acousticsProp:

Two-bladed Propeller
--------------------

With the rise of Urban Air Mobility (UAM), accurately predicting noise from propeller systems is crucial. 
CFD simulations for UAM propellers differ significantly from more conventional aircraft, involving lower tip speeds and intricate three-dimensional flow features like transitions, separations, and crossflows. 
The lower tip speeds make the prediction of broadband noise from trailing edge separation particularly important.
The complex three-dimensional flow features underscore the need for using high-fidelity CFD methods for noise prediction, raising doubts about the reliability of lower-fidelity methods like Blade Element Momentum Theory (BEMT) and Nonlinear Vortex Lattice Method (NVLM).
Precisely predicting broadband noise relies on effectively capturing turbulent fluctuations both in the boundary layer and the wake, highlighting the importance of scale-resolving approaches.

In this validation study, we explore the far-field noise signature of a two-bladed open propeller manufactured by Mejzlik, using Flow360 and its acoustic solver based on the Ffowcs-Williams and Hawkings (FWH) analogy.
The propeller, with a diameter (:math:`D_{p}`) and pitch (:math:`P`) of 9 inches (:math:`9\times9`) , was experimentally studied at the University of Bristol Aeroacoustic Facility, with results reported by `Baskaran et al. <https://doi.org/10.2514/6.2022-3108>`_ :sup:`7`.

Simulation Setup
^^^^^^^^^^^^^^^^
This study considers two operating conditions, representing hover and forward modes, with freestream velocities of :math:`0` and :math:`15 m/s`. 
The CFD domain and the specified boundary conditions are illustrated in :ref:`Fig9_Propeller_BC`.

The Reynolds numbers based on the chord length at :math:`75\%` of span (:math:`Re_{75}`) and based on the propeller's diameter (:math:`Re_{D}`) and tip Mach number (:math:`M_{tip}`) of 0.21 is :math:`Re_{75}=62816` and :math:`Re_{D}=1121855` at hover mode.
At forward mode they are: :math:`Re_{75}=64172` and :math:`Re_{D}=1146072`.

The advance ratios at hover and forward modes are :math:`\frac{U_{\infty}}{nD_{p}}=0.00` and :math:`0.66`, with the propeller rotating at :math:`6000 rpm` for both conditions.
The :math:`U_{\infty}` is the freestream velocity in :math:`m/s`, :math:`n` is the rotational speed of the propeller in revolutions per second, and :math:`D_{p}` is the propeller's diameter in :math:`m`.
The blade-passing frequency (BPF) is 200Hz, and the blade tip Mach number is :math:`0.21`. Additional freestream properties include :math:`p_{\infty} = 101325 Pa`, :math:`T_{\infty} = 288.15 K`, and :math:`\rho_{\infty} = 1.2 kg/m^{3}`.

.. _Fig9_Propeller_BC:
.. figure:: Figures/propeller_bc.png
   :align: center
   :width: 100%

   Summary of boundary conditions for the two-bladed Mejzlik propeller.

Grid Refinement Study
^^^^^^^^^^^^^^^^^^^^^
In the overall process, four unstructured grid levels are generated using a hex-dominant algorithm both in the near-field and the far-field. 
The surface mesh utilized for this validation study is a quad-dominant unstructured mesh designed to accurately capture the rounded leading-edge and the trailing-edge of the propeller geometry through anisotropic stretching.
This stretching is specifically applied to the surface based on the edge boundaries of the propeller geometry.

Four refinement zones are strategically placed around the propeller to maintain control over mesh coarsening, facilitating a gradual increase in cell edge length in the wake region.
This measured coarsening of the mesh in the wake region aids in dissipating vortical structures and preventing residual acoustic reflections that may not be sufficiently damped by non-reflective boundary conditions.
Across different grid levels, the mesh undergoes refinement in all directions, covering both the surface mesh and refinement zones.
The mesh node statistics for three mesh levels are detailed in :ref:`Tab4_Properller_Grid_Levels`, with mesh level 1 visually represented in :ref:`Fig10_Propeller_Mesh`.

.. _Tab4_Properller_Grid_Levels:
.. csv-table:: Grid point statistics for the mesh refinement study. 
   :file: Tables/propeller_tab4_grid.csv
   :widths: 15, 6, 12, 10, 13, 12
   :align: center
   :header-rows: 1
   :width: 86%

.. _Fig10_Propeller_Mesh:
.. figure:: Figures/propeller_mesh.png
   :align: center
   :width: 100%

   Grid level 1 generated for the Mejzlik propeller.

.. _Fig11_Propeller_refStudy:
.. figure:: Figures/propeller_gci.png
   :align: center
   :width: 100%

   Grid convergence study performed at hover mode for the Mejzlik propeller.

The results of the mesh convergence study are illustrated in :ref:`Fig11_Propeller_refStudy`.
For the coarse grid (level 0), 240 time steps are used per revolution, which is 3/2 degrees per time step. For the medium, fine, and extra-fine grid levels, 480 time steps per revolution (3/4 degrees per revolution) are used.
:ref:`Fig11_Propeller_refStudy` compares outcomes across four grid levels for torque, thrust, and figure of merit (FoM) against experimental values.
Concerning thrust, grid levels 2 and 3 predict it with less than a 6% error, and they forecast torque with less than a 4% error compared to experimental values. 
Notably, the low-dissipation simulation on grid level 2 yields the most accurate predictions, with a 4% error for thrust and a 1% error for torque compared to experimental values.
grid level 2 is therefore considered sufficient.

Aerodynamic Validation
^^^^^^^^^^^^^^^^^^^^^^
The aerodynamic loads at hover mode with a rotational speed of :math:`6000` rpm are compared in :ref:`Fig12_Propeller_Compare` against experimental data provided by `Baskaran et al. <https://doi.org/10.2514/6.2022-3108>`_ :sup:`7` and `Kunz et al. <https://doi.org/10.2514/6.2023-4185>`_ :sup:`8`.
In this figure, the gray curve represents the aerodynamic load curve from the experimental data. 
The magenta circle corresponds to the Delayed Detached Eddy Simulation (DDES) result on grid level 2, the cyan circle represents the Implicit Large Eddy Simulation (ILES) result without the low-dissipation scheme, and the orange circle indicates the ILES result with the low-dissipation scheme both on grid level 2.
Notably, the orange circle aligns more closely with experimental values for both thrust and torque at hover mode.
The thrust and torque values are obtained by averaging over the last 5 revolutions after an initial 10 revolutions.

.. _Fig12_Propeller_Compare:
.. figure:: Figures/propeller_valid.png
   :align: center
   :width: 100%

   Comparison of aerodynamic loads in hover mode against experimental data for ILES and DDES simulations for the Mejzlik propeller.

.. _Fig13_Propeller_History:
.. figure:: Figures/propeller_loads_history.png
   :align: center
   :width: 100%

   Comparison of the history of aerodynamic loads in hover mode for ILES and DDES simulations for the Mejzlik propeller.

In :ref:`Fig13_Propeller_History`, the history of aerodynamic loads are compared between Delayed Detached Eddy Simulation (DDES) and Implicit Large Eddy Simulation (ILES). 
For the DDES simulation, aerodynamic loads are monitored for the last 10 revolutions, while for the ILES simulation, they are monitored for the last 5 revolutions.
The DDES simulation is initiated after two consecutive initial simulations. Firstly, a first-order simulation is conducted for 5 revolutions, followed by a second-order simulation for 10 revolutions. 
Finally, the DDES simulation begins from the solution obtained in the second-order run and lasts for 10 revolutions.
The ILES simulation starts after one initial simulation. Initially, a first-order simulation is performed for 5 revolutions, and then the ILES simulation begins using the first-order solution, lasting for 10 revolutions.
The aerodynamic loads are averaged over the last 5 revolutions.
The comparison in this figure reveals that variations in torque and thrust after the initial 10 revolutions are similar. 
Therefore, it is considered that the initial 10 transient revolutions are sufficient to initiate an unsteady run and average the aerodynamic loads.

.. _Fig14_Propeller_Velocity:
.. figure:: Figures/propeller_vel_slice.png
   :align: center
   :width: 100%

   Comparison of instantaneous velocity contours in hover mode for ILES and DDES simulations for the Mejzlik propeller.

In :ref:`Fig14_Propeller_Velocity`, instantaneous velocity contours at the middle slice are presented for both Implicit Large Eddy Simulation (ILES) and Delayed Detached Eddy Simulation (DDES) approaches.
On the left side, the DDES solution on the extra fine mesh (level 3) after 25 revolutions is displayed.
This represents the longest simulation on the finest mesh. Next, the DDES solution on the fine mesh (level 2) after 15 revolutions is shown.
In comparison to the DDES solution on grid level 3, the vortices are more resolved in the near-field region, which is attributable to the presence of a finer mesh in the wake region.
Moreover, the wake is more dissipated downstream after 2D, which stems from the mesh resolution difference between level 2 and 3.
On the right side, the DDES solution on grid level 2 is compared with the ILES solution on the same grid level after 15 revolutions.
The comparison of instantaneous vortical structures for both solutions on the same grid level after the same number of revolutions reveals fair similarity.
Finally, the instantaneous ILES solution on grid level 2 after 15 revolutions is presented.
The vortical structure in the wake region for this solution is notably similar to the DDES solution on grid level 3. The vortices are well-resolved in the wake region for the low-dissipation simulation on grid level 2.
This indicates that employing the low-dissipation scheme enhances the ability to resolve turbulence in both space and time for a transient simulation.
The iso-surface of the Q-criterion for different instantaneous solutions is shown in :ref:`Fig15_Propeller_isoQ`.
It reveals some separation on the upper surface, even though this is the RANS region of the DDES, and confirms the different resolution of turbulent eddies seen earlier.

.. _Fig15_Propeller_isoQ:
.. figure:: Figures/propeller_iso_q.png
   :align: center
   :width: 100%

   Comparison of iso-surface Q-criterion in hover mode for ILES and DDES simulations for the Mejzlik propeller.

Noise Evaluation
^^^^^^^^^^^^^^^^

The acoustic field is computed at 21 observation points positioned at a distance of 1.75 meters (:math:`7.65 \times D_{p}`) from the propeller, aiming to obtain a wall-resolved directivity consistent with the experimental tests provided by `Baskaran et al. <https://doi.org/10.2514/6.2022-3108>`_ :sup:`7` and `Kunz et al. <https://doi.org/10.2514/6.2023-4185>`_ :sup:`8`.
The schematic representation of microphone locations is illustrated in :ref:`Fig16_Propeller_Observers`. 
Please note that this figure is schematic and cannot be scaled. The actual locations of the microphones are at a considerable distance from the propeller for measuring far-field noise.

.. _Fig16_Propeller_Observers:
.. figure:: Figures/propeller_mics.png
   :align: center
   :width: 60%

   Schematic locations for noise evaluation points for the Mejzlik propeller.

.. _Fig17_Propeller_Spectra:
.. figure:: Figures/mejzlik_spectra.png
   :align: center
   :width: 100%

   Sound Pressure Level (SPL) spectra in both hover and forward modes for both DDES and ILES runs of the Mejzlik propeller.

The sound pressure level (SPL) spectrum for the DDES run on grid level 3 and ILES runs on grid level 2 is shown in :ref:`Fig17_Propeller_Spectra` at microphone locations at :math:`\theta=` 50, 90, and 140 degrees, representing upstream, in-plane, and downstream observers.
As observed in this figure, the simulations accurately predict the tonal noise, with all three simulations capturing the peak and even the width of the tonal noise effectively.
The agreement between the experimental and numerical results is notably good, indicating a reliable prediction of tonal noise characteristics by the simulations.
In forward flight, there is a slight over-prediction of the tonal noise. The broadband noise is much lower, as a result of the vortices moving away from the blade much more rapidly, and/or of the blade having a lower lift coefficient and less separation.

.. _Fig18_Propeller_OASPL:
.. figure:: Figures/prop_oaspl.png
   :align: center
   :width: 100%

   Overall Sound Pressure Level (OASPL) in both hover and forward modes for both DDES and ILES runs of the Mejzlik propeller.

The Overall Sound Pressure Level (OASPL) for both hover and forward modes in both Delayed Detached Eddy Simulation (DDES) and Implicit Large Eddy Simulation (ILES) simulations is presented in :ref:`Fig18_Propeller_OASPL`.
In hover mode, the OASPL from the DDES simulation on grid level 3 closely aligns with experimental results.
Additionally, the ILES simulation with the low-dissipation scheme also exhibits a strong agreement with experimental OASPL.
In the forward mode, the ILES simulation with the low-dissipation scheme accurately predicts both the OASPL values and their directional characteristics compared to the experimental data.
This is partly thanks to compensating errors in the spectrum.
The iso-surface Q-criterion surface plot in forward mode for both DDES and ILES simulations on grid level 2 is displayed in :ref:`Fig19_Propeller_ISOFWD`. 
Some separation on the upper blade surface is seen with ILES, and probably linked to the lack of turbulence model.
This visualization highlights the improved resolution of vortical structures in the ILES simulation with the low-dissipation scheme on the same grid level.
It also shows that the DDES has no separation on the blade, unlike the ILES which lacks a fine-enough grid to resolve the boundary-layer turbulence.
Neither simulation resolves turbulence inside the tip vortex, a sign that there is much room for further grid refinement.

.. _Fig19_Propeller_ISOFWD:
.. figure:: Figures/prop_iso_fwd.png
   :align: center
   :width: 90%
   
   Iso-surface Q-criterion plot in forward mode on grid level 2 for both DDES and ILES runs of the Mejzlik propeller.

| :sup:`(1)` Jacob, M.C., Boudet, J., Casalino, D. and Michard, M., 2005. A rod-airfoil experiment as a benchmark for broadband noise modeling. Theoretical and Computational Fluid Dynamics, 19, pp.171-196.
| :sup:`(2)` Casalino, D., Jacob, M. and Roger, M., 2003. Prediction of rod-airfoil interaction noise using the Ffowcs-Williams-Hawkings analogy. AIAA journal, 41(2), pp.182-191.
| :sup:`(3)` Peth, S., Seo, J.H., Moon, Y.J., Jacob, M.C. and Thiele, F., 2006. Computation of Aerodynamic Noise from Rod Wake–Airfoil Interactions, Eur. In Conf. Comput. Fluid Dyn., The Netherlands.
| :sup:`(4)` Galdeano, S., Barré, S. and Réau, N., 2010. Noise radiated by a rod-airfoil configuration using DES and the Ffowcs-Williams & Hawkings' analogy. In 16th AIAA/CEAS aeroacoustics conference (p. 3702).
| :sup:`(5)` Giret, J.C., Sengissen, A., Moreau, S., Sanjosé, M. and Jouhaud, J.C., 2012. Prediction of the sound generated by a rod-airfoil configuration using a compressible unstructured LES solver and a FW-H analogy. In 18th AIAA/CEAS Aeroacoustics Conference (33rd AIAA Aeroacoustics Conference) (p. 2058).
| :sup:`(6)` Travin, A., Shur, M., Strelets, M. and Spalart, P., 2000. Detached-eddy simulations past a circular cylinder. Flow, turbulence and combustion, 63(1-4), pp.293-313.
| :sup:`(7)` Baskaran, K., Jamaluddin, N.S., Celik, A., Rezgui, D. and Azarpeyvand, M., 2022. Aerodynamic and aeroacoustic characteristics of propellers with different blade numbers. In 28th AIAA/CEAS Aeroacoustics 2022 Conference (p. 3108).
| :sup:`(8)` Kunz, F., Zhou, B.Y., Pullin, S.F., Azarpeyvand, M., Galimberti, L., Morelli, M. and Guardone, A., 2023. High-Fidelity Propeller Broadband Noise Prediction using SU2. In AIAA AVIATION 2023 Forum (p. 4185).

Conclusions
-----------

The validation study provides several recommendations for aeroacoustics and noise simulation:

1. **Extruded Configuration and Spanwise Length**:

   - For extruded configurations, it is advisable to set the spanwise length close to or larger than the spanwise coherence length to effectively resolve flow structures. This would be difficult for the coherence length at the vortex-shedding frequency.
   - Using periodicity is recommended, but when employing CFD with the FWH analogy, it may be necessary to distribute observer locations in the span direction to account for length effects. This is not an issue with real-world systems which are finite in all directions.

2. **Reynolds Number Considerations**:

   - For low Reynolds number flow regimes with transitional unsteady flow, Implicit Large-Eddy Simulation (LES) is recommended, because the RANS modeling in DDES treats the boundary layer as turbulent everywhere.
   - For high Reynolds number flows, Delayed Detached Eddy Simulation (DDES) is recommended because RANS modeling is appropriate.

3. **Assurance of CFD Simulation Quality**:

   - It is crucial to ensure that the CFD simulation used for aeroacoustic analysis is of high quality and effectively captures unsteady loads.

4. **Mesh Resolution**:

   - Adequate mesh resolution on solid bodies, near-field and wake regions, and around noise sources is essential for the accurate capture of Sound Pressure Level (SPL) spectra and Overall Sound Pressure Level (OASPL).
   - Performing a mesh convergence study is recommended to achieve optimal mesh resolution with desirable accuracy and computing cost.

5. **Low-Dissipation Scheme**:

   - The use of a low-dissipation scheme is advised as it enhances turbulence resolution in both time and space, particularly on relatively coarser meshes, resulting in more accurate SPL spectra and OASPL.

6. **Validation of Flow360 Solver**:

   -  The Flow360 solver has been validated against experimental data and other scale-resolving simulations for two test cases: laminar separation flow over the rod-airfoil configuration and a two-bladed propeller.
   -  Good agreement is observed for noise spectra and OASPL when compared to experimental data.

These recommendations highlight key considerations and best practices for aeroacoustic analysis using Flow360.
