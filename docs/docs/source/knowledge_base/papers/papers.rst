.. _papers:

Papers
======

.. _Fundamental_Studies_Rotor:

Fundamental Studies Towards Rotor Simulations and Design
---------------------------------------------------------------------------------------------------

| Feilin Jia and Philippe Spalart
| *Flexcompute Inc, Belmont, Massachusetts, 02138*


| Maks J Groom and Qiqi Wang
| *Massachusetts Institute of Technology, Cambridge, Massachusetts 02139*
| *Flexcompute Inc, Belmont, Massachusetts, 02138*

We use numerical solutions to investigate the actuator-disc model in hover. Most solutions come from a conventional Navier-Stokes solver named Flow360, and a few from a Vortex-Ring Method. In Flow360 the source term for vorticity, namely the curl of the force field, is regularized and we have failed to obtain steady solutions with very tight regularizations; the vortex sheet becomes so thin that Kelvin-Helmholtz waves appear, which we consider physically correct. The findings agree with theory and earlier numerical work, and not with the textbook presentations or intuition, and that in three respects. First, even with a uniform pressure jump on a flat disc the velocity through the disc is far from uniform, and it is directed upwards near the rim. The cross-section of the vortex sheet begins as a 45 |deg| spiral and therefore it has an initial excursion over the disc, as opposed to a funnel starting downwards with low curvature. Second, again with uniform load, the entire flow field depends only on the rim of the disc, and not on the shape of the surface; only the pressure field reacts to a change in shape (the fluid is incompressible). This opens the door to drooped disc edges/blade tips in actual designs, which could control blade-vortex interaction and also have a "winglet effect" if the rotor diameter is constrained. Third, according to theory, uniform loading gives the lowest power consumption for a given thrust and diameter, in other words the best Figure of Merit, even though the velocity through the disc is so non-uniform; this is because the velocity far downstream is uniform, and that is what controls the power. It is important to invalidate misleading textbook assertions, because astute users will question the results of CFD when they disagree with these assertions. We however find out numerically and from both methods that some non-uniform distributions (higher towards the tip) give a Figure of Merit slightly superior to that from the uniform load; this needs to be explored.

`Full Paper <https://simcloud-public-1.s3.amazonaws.com/publications/Jia_et_al_2023_Fundamental_Studies_Towards_Rotor_Simulations_and_Design.pdf>`__

.. _Hover_Prediction:

An Application of the Flow360 Solver to the Hover Download Prediction Problem
---------------------------------------------------------------------------------------------------

| Thomas Fitzgibbon, CJ Doolittle and Philippe Spalart
| *Flexcompute Inc, Belmont, Massachusetts, 02138*

| Qiqi Wang
| *Massachusetts Institute of Technology, Cambridge, Massachusetts 02139*
| *Flexcompute Inc, Belmont, Massachusetts, 02138*

This paper presents the contribution of Flexcompute to the Hover Prediction Workshop, with a focus on hover performance and download predictions. The analysis is focused on HVAB rotor blade simulations at two blade tip Mach numbers of 0.58 and 0.65. First, a rigorous mesh refinement and time step study is performed to assess the discretization error sensitivities for both isolated and installed rotors. The impact of mesh resolution and time step on the performance, sectional loading and flow features is presented with recommendations put forward for engineering level accuracy and high-fidelity solutions. An analysis of loads convergence is also performed, which was found to affect the predictions for installed rotor calculations. Next, a collective sweep study is performed at two blade tip Mach numbers for isolated and installed rotors, and where available comparisons are made with experimental data and predictions from other CFD codes. The Flow360 results showed strong correlation with reference data and resolved high-resolution wake structures, showing the applicability of Flow360 to hovering rotor solutions.

`Full Paper <https://simcloud-public-1.s3.amazonaws.com/publications/Fitzgibbon_et_al_2023_An_Application_of_the_Flow360_Solver_to_the_Hover_Download_Prediction_Problem.pdf>`__

.. _HLPW4Paper:

An Analysis of Modeling Sensitivity Effects for High Lift Predictions using the Flow360 CFD Solver
---------------------------------------------------------------------------------------------------

| Thomas Fitzgibbon, Philippe Spalart and Jim Bungener
| *Flexcompute Inc, Belmont, Massachusetts, 02138*

| Qiqi Wang
| *Massachusetts Institute of Technology, Cambridge, Massachusetts 02139*
| *Flexcompute Inc, Belmont, Massachusetts, 02138*

This paper presents the contribution of Flexcompute to the 4th High Lift Prediction Work-
shop based on the Flow360 solver. The analysis of the high-lift prediction results is focused on
four key areas: mesh sensitivity, turbulence modelling sensitivity, effect of cold- vs warm-starts
and an examination of hysteresis effects. The mesh sensitivity study includes the use of different
mesh topologies, families and levels of mesh refinement for the committee and non-committee
provided grids. Through the analysis of the resolution of the different flow features recommen-
dations are put forward regarding meshing for future workshops. Next, results are presented
using different RANS-based turbulence models, with a focus on the eddy viscosity levels in the
flow structures and separated flow predictions. Following, a detailed analysis is performed for
comparing cold-starts versus warm-starts with a focus on comparing the differences in the skin
friction and flow fields at each angle of attack. Hysteresis effects are also examined to estab-
lish the effect of the initial condition on the solution. Finally, the best practice RANS results
are analyzed in detail along with DES simulations with the aim to provide conclusions of the
ability of RANS to predict high lift flows. The DES results were found to significantly improve
the comparison with experimental data and showed high confidence in terms of achieving the
correct answer for the right reasons.

`Full Paper <https://simcloud-public-1.s3.amazonaws.com/publications/HLPW4/HLPW_Paper.pdf>`__

`Slides <https://simcloud-public-1.s3.amazonaws.com/publications/HLPW4/HLPW_In_Person_Presentation.pdf>`__

.. _propModelingApproach:

Impact of the Propulsion Modeling Approach on High-Lift Force Predictions of Propeller-Blown Wings
---------------------------------------------------------------------------------------------------

| Cecile Casses and Chris Courtin
| *Electra.aero, Falls Church, 22042*

| Mark Drela
| *Massachusetts Institute of Technology, Cambridge, Massachusetts 02139*

| Thomas Fitzgibbon, Runda Ji, Maciej Skarysz and Philippe Spalart
| *Flexcompute Inc, Belmont, Massachusetts, 02138*

| Qiqi Wang
| *Massachusetts Institute of Technology, Cambridge, Massachusetts 02139*
| *Flexcompute Inc, Belmont, Massachusetts, 02138*

Distributed electric propulsion presents new opportunities to design aircraft which take
advantage of the deliberate close interaction between propellers, wings, and flaps. Significant
forces and moments arise which would not be well-captured by modeling the systems in isolation.
The ability to accurately predict these forces is important for a number of new vehicle designs,
including electric short takeoff and landing (eSTOL) aircraft which use the interaction between
the propellers and wing/flaps to generate very high wing lift coefficients, if normalized by the
flight speed. The flow field at the propellers is strongly influenced by the presence of the wing,
and vice versa.
This paper presents CFD analyses relevant to an eSTOL wing in a high-lift configuration,
comparing four different approaches to modeling the propellers - an actuator disk, a steady and
an unsteady blade element model, and a blade-resolved unsteady simulation. This is done in
the Flow360 Navier-Stokes solver developed by Flexcompute. We begin with an isolated rotor,
continue with a model problem that comprises a single rotor with behind it a section of wing
and flap, and conclude with a full 3D configuration. The actuator disk is advantageous from
a computational time point of view, but introduces error into the solutions because it cannot
adapt to strongly varying local inflow conditions. The time-resolved simulations are expected to
give the most accurate solutions, but are computationally expensive. An adaptive blade-element
model gives a good compromise between accuracy and performance.

`Full Paper <https://simcloud-public-1.s3.amazonaws.com/publications/Electra_Joint_Paper/Electra_FC_Impact_of_Propulsion_Modeling_Approach_on_High_Lift_Force_Prediction_of_Propeller_Blown_Wings.pdf>`__

`Slides <https://simcloud-public-1.s3.amazonaws.com/publications/Electra_Joint_Paper/Electra_AIAA_2022_Presentation.pdf>`__

Flexcompute Contribution to the VIIth AIAA Drag Prediction Workshop
-------------------------------------------------------------------

| Thomas Fitzgibbon and Philippe Spalart
| *Flexcompute Inc, Belmont, Massachusetts, 02138*

| Qiqi Wang
| *Massachusetts Institute of Technology, Cambridge, Massachusetts 02139*
| *Flexcompute Inc, Belmont, Massachusetts, 02138*

This presentation presents the VIIth AIAA Drag Prediction Workshop cases results using the Flow360 solver. This includes a mesh convergence study, alpha sweep with static aeroelastic deflections and a Reynolds number study for the CRM wing-body configuration in transonic flow conditions. Comparisons are made with experimental data where available.

`Slides <https://simcloud-public-1.s3.amazonaws.com/publications/DPW7/DPW7_AIAA_2022_Presentation.pdf>`__

.. _XV15_BET:

XV-15 Rotor Simulation in Flow360 using the Blade Element Theory
---------------------------------------------------------------------------------------------------

| John Moore and Feilin Jia
| *Flexcompute Inc, Belmont, Massachusetts, 02138*

| Qiqi Wang
| *Massachusetts Institute of Technology, Cambridge, Massachusetts 02139*
| *Flexcompute Inc, Belmont, Massachusetts, 02138*

This paper presents results of the numerical study of the Bell XV-15 rotor with the flow solver Flow360 using a coupled Navier-Stokes/Blade Element Theory model. Results are presented for both steady-state and transient simulations. The resulting thrust, torque, and blade loadings show good agreement with high-fidelity DDES results computed with Flow360 and experiments.

`Full Paper <https://simcloud-public-1.s3.amazonaws.com/publications/XV-15_Rotor_Simulation_in_Flow360_using_the_Blade_Element_Theory.pdf>`__

.. _rotor5Paper:

Rotor5: Rotor analysis under 5 hours using ultra-fast and high-fidelity CFD simulation and automatic meshing
------------------------------------------------------------------------------------------------------------

| Runda Ji, Feilin Jia, Philippe Spalart and Zongfu Yu
| *Flexcompute Inc, Belmont, Massachusetts, 02138*

| Qiqi Wang
| *Massachusetts Institute of Technology, Cambridge, Massachusetts 02139*
| *Flexcompute Inc, Belmont, Massachusetts, 02138*


We introduce a novel workflow called Rotor5 to simplify and accelerate the traditional high-fidelity
rotor simulations by integrating (1) CAD preparation (2) mesh generation and (3) CFD solver into an
end-to-end process. We quantify the limitations of a popular low-fidelity rotor design tool called Xrotor
and demonstrate the necessity of using a high-fidelity CFD solver such as Rotor5. Using both Xrotor and
Rotor5, we investigate the tiltrotor XV-15 at two different flight conditions: (1) airplane propeller mode
in forward flight and (2) helicopter hovering mode, where the fundamental limitations of low-fidelity
Xrotor could cause a catastrophic design failure. A major cause for this is the tip vortex of the preceding
blade.

`Full Paper <https://simcloud-public-1.s3.amazonaws.com/publications/Rotor5_arXiv.pdf?download=false>`__

`Slides <https://simcloud-public-1.s3.amazonaws.com/publications/Rotor5_VFS_Presentation.pdf?download=false>`__

.. _DESXV15:

Assessment of Detached Eddy Simulation and Sliding Mesh Interface in Predicting Tiltrotor Performance in Helicopter and Airplane Modes
--------------------------------------------------------------------------------------------------------------------------------------

| Feilin Jia and John Moore
| *Flexcompute Inc, Belmont, Massachusetts, 02138*

| Qiqi Wang
| *Massachusetts Institute of Technology, Cambridge, Massachusetts 02139*

This paper presents numerical investigation on performance and flow field of the full-scale XV-15 tiltrotor in both helicopter mode (hovering flight and forward flight) and aeroplane propeller mode using Detached Eddy Simulation, in which the movement of the rotor is achieved using a Sliding Mesh Interface. Comparison of our CFD results against experiment data and other CFD results is performed and presented.

`Full Paper <https://simcloud-public-1.s3.amazonaws.com/publications/Jia_et_al_2022_Assessment_of_Detached_Eddy_Simulation_and_Sliding_Mesh_Interface_in_Predicting_Tiltrotor_Performance_in_Helicopter_and_Airplane_Modes.pdf>`__

.. _aeroParametricAdaptive:

Aerodynamic Risk Assessment using Parametric, Three-Dimensional Unstructured, High-Fidelity CFD and Adaptive Sampling
---------------------------------------------------------------------------------------------------------------------

| Runda Ji
| *Flexcompute Inc, Belmont, Massachusetts, 02138*

| Qiqi Wang
| *Massachusetts Institute of Technology, Cambridge, Massachusetts 02139*

We demonstrate an adaptive sampling approach for computing the probability of a rare event for a set of three-dimensional airplane geometries under various flight conditions. We develop a fully automated method to generate parameterized airplanes geometries and create volumetric mesh for viscous CFD solution. With the automatic geometry and meshing, we perform the adaptive sampling procedure to compute the probability of the rare event. We show that the computational cost of our adaptive sampling approach is hundreds of times lower than a brute-force Monte Carlo method.

`Full Paper <https://simcloud-public-1.s3.amazonaws.com/publications/Ji_et_al_2021_Aerodynamic_Risk_Assessment_using_Parametric_Three-Dimensional_Unstructured_High-Fidelity_CFD_and_Adaptive_Sampling.pdf>`__

