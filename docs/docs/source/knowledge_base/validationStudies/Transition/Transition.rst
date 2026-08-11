.. _transitionValidationStudy:
.. |agr|  unicode:: U+003B1 .. GREEK SMALL LETTER ALPHA
.. |omega|    unicode:: U+03C9 .. OMEGA SIGN


Transition Modeling
===================

Introduction
------------

The main purpose of this validation study is verification and validation of the transition model implemented within the Flow360 solver. Additionally, mesh sensitivity effects are examined to provide best-practice guidelines for using the transition modelling capability in the Flow360 solver. Transition modelling is of key importance when a significant portion of the boundary layer remains laminar and the fully-turbulent modelling assumption is no longer valid. Transition to turbulence occurs at low Reynolds numbers (order of :math:`10^5` and lower), and is important to capture in many aerospace applications such as turbomachinary, rotorcraft and fixed-wing aircraft. The model implemented in the Flow360 solver is based on the the SA-AFT (Amplification Factor Transport) 2019b model developed by Coder with details of the model presented on `the NASA Turbulence Modelling Resource (TMR) Website <https://turbmodels.larc.nasa.gov/aft_transition_3eqn.html>`_. The AFT model is also compatible with the :math:`k-\omega` SST turbulence model. The AFT model is based on linear stability theory, which aims to track the growth of pressure/velocity instabilities in the boundary layer. This approach more directly aligns with the physics involved with the transition process as compared to correlation based models. The model is coupled with the Spalart-Allmaras (SA) turbulence model through a modification of the :math:`f_{t2}` term with two additional transport equations solved, one for modified intermittency and one for the approximate envelope amplification factor. 

The present validation study targets two cases, a canonical zero-pressure gradient flat plate and flow past an NLF(1)-0416 airfoil, designed to maintain laminar flow in aerospace applications. Firstly, the zero-pressure gradient flat plate case is demonstrated with transition based on the `NASA TMR Flat Plate Validation Case <https://turbmodels.larc.nasa.gov/flatplate.html>`_. The cases from the `1st Transition Prediction Workshop <https://transitionmodeling.larc.nasa.gov/workshop_i/>`_ are not used here, as transition models based on the AFT framework are not tailored to cases with high freestream turbulent eddy viscosity ratio's, which target bypass transition conditions. An assessment of grid element type is made to examine mesh sensitivity effects. Following, detailed comparisons are made with the results presented by `Coder <https://arc.aiaa.org/doi/10.2514/6.2019-0039>`_ using the SA-AFT model in OVERFLOW. The primary objective of this case is verification of the SA-AFT model in the Flow360 solver, as well as preliminary validation and examination of mesh sensitivity effects.

Next, results are presented for the NLF(1)-0416 airfoil. A detailed mesh sensitivity study is presented that includes an examination of 2D vs pseudo-3D effects, grid element type and topology, with the aims to provide best-practices when using the Flow360 solver. A simulation is also performed for a 3D configuration to confirm the validity of the pseudo-3D results. Then, results from the mesh convergence study are compared with `data from Coder <https://arc.aiaa.org/doi/10.2514/6.2019-0039>`_ using the SA-AFT model in OVERFLOW and `Venkatachari et al. <https://arc.aiaa.org/doi/abs/10.2514/6.2022-3679>`_ who presented data using the :math:`\gamma- Re_\theta` transition model in OVERFLOW and FUN3D. Additionally, results obtained for the full angle of attack sweep are compared with `experimental data <https://ntrs.nasa.gov/citations/19810015487>`_ and OVERFLOW data from `Coder <https://arc.aiaa.org/doi/10.2514/6.2019-0039>`_ with the aim to validate the Flow360 transition model.

Flat Plate Case
---------------


Simulation Setup
^^^^^^^^^^^^^^^^

The flat plate case is based on the `model verification case available on the NASA TMR website <https://turbmodels.larc.nasa.gov/flatplate.html>`_. The freestream Mach number is equal to 0.2 with the Reynolds number per unit length equal to 5 million. The boundary conditions applied are based on the Flat Plate TMR Case as shown in :ref:`Fig1_FlatPlate_BoundaryConditions`.

.. _Fig1_FlatPlate_BoundaryConditions:
.. figure:: Figures/FlatPlate_BoundaryConditions.png
   :align: center
   :width: 80%

   Summary of boundary conditions and operating conditions for the zero pressure gradient flat plate case.

The simulations are performed using SA-AFT model with an :math:`N_{crit}` value of 10.3 for consistency with the study of `Coder <https://arc.aiaa.org/doi/10.2514/6.2019-0039>`_ and data from `experiments <https://ntrs.nasa.gov/citations/19930092285>`_ for the skin friction coefficient. The grids were taken from the `NASA TMR Flat Plate grids download page <https://turbmodels.larc.nasa.gov/flatplate_grids.html>`_ with five grid levels available ranging from 35 by 25 to 545 by 385 in the x-z plane with 2 nodes in the spanwise direction, as shown in :ref:`Tab1_GridRef_FlatPlate`.

.. _Tab1_GridRef_FlatPlate:
.. csv-table:: Grid dimensions for the flat plate case mesh convergence study. 
   :file: Tables/flat_plate_tab1_gridref.csv
   :widths: 20, 50
   :align: center
   :header-rows: 1
   :delim: @

The five grid levels are used to conduct a mesh convergence study and assess what impact the mesh refinement has on the transition location. Additionally, grids with different element types are also compared to perform an initial mesh sensitivity study. The baseline hexahedral grid was split into prisms to give a triangulated grid on the x-z plane. Note as the grid has only 2 nodes across, quadrilaterals are still used in the spanwise direction and hence are on the surface of the flat plate. A comparison of the mid-span (x-z) plane is shown in :ref:`Fig2_Flat_Plate_Grid_Comparison`.

.. _Fig2_Flat_Plate_Grid_Comparison:
.. figure:: Figures/Flat_Plate_TriVsQuad.png
   :align: center
   :width: 80%

   Comparison of the quad (top) and tri (bottom) grids for the flat plate case.

Numerical Results
^^^^^^^^^^^^^^^^^

The mesh convergence for the flat plate case for quad and tri meshes is analysed in terms of the skin friction distribution shown in :ref:`Fig6_Flat_Plate_Skin_Friction`.

.. _Fig6_Flat_Plate_Skin_Friction:
.. figure:: Figures/Flat_Plate_Skin_Friction_Grid_sensitivity.png
   :align: center
   :width: 99%

   Comparison of skin friction distribution grid convergence for quad and tri meshes for the flat plate case.

The grid convergence of the skin friction distribution on the flat plate shows that the result is not fully grid converged. The behaviour of the meshes with quads is different than the meshes with tri elements. For the coarsest tri mesh, no transition is detected and as the grid is refined, the transition location moves upstream, apart for the finest two levels where the transition location moves downstream. For the quad meshes, the convergence is highly non-linear. Initially the transition location moves downstream, before moving upstream for grid level 3, and then moving downstream for levels 4 and 5. The transition location for the finest two meshes (level 5) is very close for both quad and tri meshes. The tri meshes appear to be more dissipative for coarser meshes, however, have a lower sensitivity to transition location for finer meshes when compared to the quad meshes. Next the skin friction distributions are compared for the two finest grids with data from `OVERFLOW <https://arc.aiaa.org/doi/10.2514/6.2019-0039>`_ and experimental data from `Schubauer-Klebanoff <https://ntrs.nasa.gov/citations/19930092285>`_, shown in :ref:`Fig7_Flat_Plate_Skin_Friction2`.

.. _Fig7_Flat_Plate_Skin_Friction2:
.. figure:: Figures/Flat_Plate_Skin_Friction_Experimental.png
   :align: center
   :width: 99%

   Comparison of skin friction distribution for the two finest grids with OVERFLOW and experimental data for the flat plate case.

The skin friction distributions indicate that both Flow360 and OVERFLOW predict a transition location that is further downstream when compared to experiments. Flow360 predicts a transition location slightly further downstream than OVERFLOW, however, the difference reduces as the grid is refined. Finer grids would be required to perform a complete verification of the transition model in Flow360. The grid convergence for the finest two grids shows the same pattern between Flow360 and OVERFLOW, as the L5 grid moves further away from experiments than the L4 grid. The lack of grid convergence for the flat plate was due to the intermittency equation and its coupling with the SA turbulence model, according to Coder. Another potential aspect is the fact that the transition location may be highly sensitive to numerics with no presence of a pressure gradient. To further examine the differences between Flow360 and OVERFLOW, the volumetric modified intermittency and amplification factor contours are extracted from the solutions, shown in :ref:`Fig8_Gamma` - :ref:`Fig9_NHat`.

.. _Fig8_Gamma:
.. figure:: Figures/Gamma.png
   :align: center
   :width: 99%

   Comparison of the modified intermittency contours between Flow360 and OVERFLOW for flat plate case on the finest grid.

.. _Fig9_NHat:
.. figure:: Figures/NHat.png
   :align: center
   :width: 99%

   Comparison of the amplification factor contours between Flow360 and OVERFLOW for flat plate case on the finest grid.

The contours of the modified intermittency show a high degree of consistency between Flow360 and OVERFLOW. The primary difference is the prediction of the transition location slightly further downstream for the Flow360 solution. The Flow360 prediction also predicts a slightly larger turbulent spot ahead of the transition location. The amplification factor contours also show good correlation, although slightly higher magnitudes are observed in the Flow360 solutions. This is primarily due to the prediction of transition further downstream, hence the instabilities have a longer distance to grow. The tri and quad solutions were consistent with differences mainly attributed to the different transition location.
   
   
NLF(1)-0416 Airfoil Case
------------------------

The second case that was analysed was the NLF(1)-0416 airfoil case, which is more applicable to aerospace applications. Firstly, the simulation setup is presented including a number of grid families generated to examine mesh sensitivity effects. The first part of the results focuses on an analysis of grid convergence for each grid family is performed, before further analysing the effect of 2D vs pseudo-3D solutions and the effect of element types. In the final part of the results, the solutions are compared with other solvers and experimental data.

Simulation Setup
^^^^^^^^^^^^^^^^

The NLF(1)-0416 airfoil case is based on `Case 2 of the 1st Transition Prediction Workshop <https://transitionmodeling.larc.nasa.gov/wp-content/uploads/sites/109/2020/02/TransitionMPW_CaseDescriptions.pdf>`_. The flow conditions are a Mach number of 0.1, Reynolds number of 4 million based on the chord length and freestream :math:`N_{crit}` value of 7.2. For this case, the SA-AFT transition model is also used. The case involves a mesh convergence study at an angle of attack of 0 and 5 degrees as well as a full angle of attack sweep.

Grid Families
.............

As mesh sensitivity effects are of interest to provide best-practice guidelines, multiple grid families are generated for the mesh convergence study. The primary aims are to establish whether 2D simulations are valid when compared to pseudo-3D and 3D simulations for transition predictions, and what element types should be used on the surface and in the volume mesh to reduce grid sensitivity effects. For this purpose, the following grid families are generated, with further details provided below:

#. **Family 1**: 2D baseline hexahedral grids (6 levels) provided on the `1st Transition Prediction Workshop Page <https://transitionmodeling.larc.nasa.gov/workshop_i>`_ . These were modified to only include 2 nodes (rather than 3) in the spanwise direction for pure 2D solutions.

#. **Family 2**: Pseudo-3D hexahedral grids. Modified baseline hexahedral grids to include 20 nodes in the spanwise direction, generated through spanwise extrusion

#. **Family 3**: 2D triangulated baseline grids. The cells on the symmetry planes were triangulated, similarly as for the flat plate validation study. This leads to quadrilateral cells on the airfoil surface with prism cells in the volume, with 2 nodes in the spanwise direction.

#. **Family 4**: Pseudo-3D triangulated baseline grids. Modified 2D triangulated baseline grids (Family 3) to include 20 nodes in the spanwise direction, generated through spanwise extrusion

#. **Family 5**: Pseudo-3D triangulated surface baseline grids. These were generated manually by applying the same number of nodes and LE/TE spacings as for the Family 2 grids and generating a triangulated structured surface grid, and then generating an unstructured prism-tet volume grid with settings extracted from the Family 1 grids. These grids have 20 nodes in the spanwise direction (over 1 chord length).

#. **Family 6**: Pseudo-3D triangulated surface baseline grids with varying number of spanwise nodes. This family includes the ultra-fine grid (Level 6) settings from Family 5 with spanwise node values of 20, 40 and 80.

#. **Family 7**: Pseudo-3D fully unstructured grids. These grids were generated using the automated meshing workflow of Flow360. The inputs to the surface and volume mesher were extracted from the Family 1 grids to ensure a high degree of consistency between different grid families. To minimize the node count, the span is reduced from 1 chord to 0.04 chord.

Grid Levels
...........

Families 1 to 4 are all based on the baseline C-topology committee grids. The baseline committee extracted grid properties (approximated) are shown in :ref:`Tab2_GridRef_NLF`. 

.. _Tab2_GridRef_NLF:
.. csv-table:: Approximate extracted grid properties for the NLF(1)-0416 case mesh convergence study. :math:`\Delta S` = Spacing on the surface, :math:`\Delta Y` = First cell height in the volume, GR = Growth Rate, C = Chordwise, N = Normal, W = Wake Cut (downstream of TE).
   :file: Tables/NLF_tab2_gridref.csv
   :widths: 20, 40, 40, 40, 40, 30, 50
   :align: center
   :header-rows: 1
   :delim: @

Families 1 and 2 have hexahedrals in the volume grid, whereas Families 3 and 4 have prisms in the volume grid, with all four families having quadrilateral elements on the surface. Families 5, 6 and 7 have triangular elements on the surface with prisms in the volume grid, and were generated from an ESP model of the NLF(1)-0416 airfoil with a blunt trailing edge. The main difference between Families 5, 6 and 7 is the fact that the automated meshing workflow does not guarantee an equal spacing in the spanwise direction on the surface. Grids in Family 7 will therefore, have more nodes in the spanwise direction near the leading and trailing edges than in the airfoil midsection, primarily due to the constraint on the cell aspect ratio of 25 (for this reason the span was reduced from 1 to 0.04), whereas grids in Family 5 and 6 have a uniform spanwise spacing along the airfoil. The settings for the surface meshes in Family 7 and the volume meshes in Families 5-7 were exactly the same as shown in the table above. The total number of nodes for Families 1-7 are shown in :ref:`Tab3_GridRef_NLF2`. Note that L6 in Family 5 is the same grid as L1 in Family 6 (20 nodes in spanwise direction).

.. _Tab3_GridRef_NLF2:
.. csv-table:: Number of nodes for the various grid levels in Families 1 to 7.
   :file: Tables/NLF2_tab3_gridref.csv
   :widths: 20, 30, 30, 30, 30, 30
   :align: center
   :header-rows: 1
   :delim: @

As can be seen from the table above, the pseudo-3D cases require significantly more grid points than the pure 2D cases. The triangular surface grids (Families 5-7) require fewer points for a given grid level primarily due to the fact that they were generated using an unstructured solver, not spanwise extrusion, hence the number of spanwise nodes in the volume typically reduces closer to the farfield. Grid Family 6 contains an increase in spanwise nodes up to 80 for level 3. A grid of 40 nodes in the spanwise direction was also generated for the grid Family 2 but not shown in the Table (25,025,088 nodes). Grid Family 7 contains significantly more nodes than Family 5 even though the spanwise spacing was reduced to 0.04, which is primarily due to the aspect ratio constraint of 25 for these grids. In the mesh convergence study, the metric used along the x-axis is 1 over the number of surface nodes. As the number of chordwise nodes will be more important than the number of spanwise nodes, the number of surface nodes is divided by 20 for pseudo-3D cases (to align Families 1 and 2, 3 and 4 on the x-axis). 

Grid Comparisons
................

The surface grids and cuts through the mid-plane of the volume mesh are presented for Level 2 grids of different Families in :ref:`Fig3_NLF_Grid_Comparison` and :ref:`Fig4_NLF2_Grid_Comparison`. The Family 6 grids are not shown here, as these used the L6 grids of Family 5 with a variation in the number of spanwise nodes.

.. _Fig3_NLF_Grid_Comparison:
.. figure:: Figures/NLF_Grid_Comparison1.png
   :align: center
   :width: 99%

   Comparison of the surface grids for different grid families.

.. _Fig4_NLF2_Grid_Comparison:
.. figure:: Figures/NLF_Grid_Comparison2.png
   :align: center
   :width: 99%

   Comparison of the volume grids for different grid families (slice through mid-span y=-0.5).

It is clear from the volume grid visualizations that the grid Family 7 is a much higher quality unstructured grid than grid Family 5 though a higher number of nodes was used. This is primarily due to the fact the grid Family 5 has 20 nodes across a spanwise length of 1 chord, leading to high aspect ratio cells. Furthermore, there is little control over the volume grid in the mid section of the airfoil.

Finally, a single grid was also generated for the 3D wing geometry using an aspect ratio of 20 (semi-span of 10 chords). The grid was generated using the automated meshing workflow of Flow360. The cell aspect ratio constraint of 25 was loosened to 100 to reduce the number of nodes in the grid. The surface settings were based on the L2 grid properties for the surface and the L6 grid properties for the volume. The primary purpose of the 3D wing simulation is to ensure that similar transition locations were obtained as for the pseudo-3D cases away from the root and tip of the wing. To perform these simulations, the :ref:`user defined dynamics <user_defined_dynamic_user_guide>` feature was switched on with the wing CL trimmed to the values obtained for the L6 grid of grid Family 1. This is done by automatically adjusting the angle of attack, according to an update law specified by the user to match the specified lift coefficient. A section of the surface grid as well as a slice at mid-span of the volume grid for the 3D wing configuration is shown in :ref:`Fig5_NLF2_Grid_Comparison`

.. _Fig5_NLF2_Grid_Comparison:
.. figure:: Figures/NLF_Grid_Comparison3.png
   :align: center
   :width: 99%

   A section of the surface grid along with the volume grid (slice through mid-span y=-0.5) for the 3D wing configuration.


Grid Convergence Study
^^^^^^^^^^^^^^^^^^^^^^

Firstly, the convergence of the integrated loads with mesh refinement is extracted at two angles of attack of 0 and 5 degrees shown in :ref:`Fig9_MeshConvergenceCL`-:ref:`Fig10_MeshConvergenceCD2`. Note that the number of surface nodes was divided by 20 for all pseudo-3D cases to ensure a fair comparison (number of nodes in spanwise direction less important than in chordwise direction).

.. _Fig9_MeshConvergenceCL:
.. figure:: Figures/CL0.png
   :align: center
   :width: 99%

   Convergence of the lift coefficient with mesh refinement at :math:`\alpha = 0^o` for 7 different grid families.


.. _Fig9_MeshConvergenceCL2:
.. figure:: Figures/CL5.png
   :align: center
   :width: 99%

   Convergence of the lift coefficient with mesh refinement at :math:`\alpha = 5^o` for 7 different grid families.


.. _Fig10_MeshConvergenceCD:
.. figure:: Figures/CD0.png
   :align: center
   :width: 99%

   Convergence of the drag coefficient with mesh refinement at :math:`\alpha = 0^o` for 7 different grid families.

.. _Fig10_MeshConvergenceCD2:
.. figure:: Figures/CD5.png
   :align: center
   :width: 99%

   Convergence of the drag coefficient with mesh refinement at :math:`\alpha = 5^o` for 7 different grid families.



The lift and drag convergence curves indicate a certain degree of non-linearity with mesh refinement for a number of grid families. Families 1-4 appear to converge towards a single value for both lift and drag at both angles of attack for the finest grid levels, however, even finer grids than presented here are required for convergence of the integrated loads. At first glance, the effect of 2D vs pseudo-3D is minimal with the tri meshes showing a lower grid sensitivity (especially from level 2 onwards). Family 5 shows significantly different convergence curves when compared to the other grid families with a high lift and drag prediction and low sensitivity with mesh refinement. Family 6 indicates at first glance that adding nodes in the spanwise direction to the Family 5 grids, aligns the integrated loads with the other grid families. The automated meshing unstructured grids show a lower variation with mesh refinement, although the node counts for these grids is higher than for other mesh families. The reasons for these effects are examined in more detail in the following figures. Firstly, however, the grid convergence of each mesh family is assessed by extracting the skin friction distributions across different grid levels shown in :ref:`Fig11_CfConvergenceFamily1`-:ref:`Fig14_CfConvergenceFamily7`.

.. _Fig11_CfConvergenceFamily1:
.. figure:: Figures/CF_Convergence_Family1.png
   :align: center
   :width: 99%

   Convergence of the skin friction distribution with mesh refinement at two angles of attack for grid Family 1.

.. _Fig12_CfConvergenceFamily3:
.. figure:: Figures/CF_Convergence_Family3.png
   :align: center
   :width: 99%

   Convergence of the skin friction distribution with mesh refinement at two angles of attack for grid Family 3.

.. _Fig13_CfConvergenceFamily5:
.. figure:: Figures/CF_Convergence_Family5.png
   :align: center
   :width: 99%

   Convergence of the skin friction distribution with mesh refinement at two angles of attack for grid Family 5.

.. _Fig14_CfConvergenceFamily7:
.. figure:: Figures/CF_Convergence_Family7.png
   :align: center
   :width: 99%

   Convergence of the skin friction distribution with mesh refinement at two angles of attack for grid Family 7.

The skin friction distributions for grid Family 1 show signs of inadequate grid convergence as the transition locations are still sensitive to mesh resolution. At both angles of attack the transition locations shift aft with mesh refinement. The upper surface at an angle of attack of 0 degrees and lower surface at 5 degrees, show signs of mesh convergence. The transition locations on the lower surface at 0 degrees and upper surface at 5 degrees, continue to move aft showing the requirement for even finer meshes. Grid Family 3 shows similar mesh convergence compared to grid Family 1 for the finest grids (L4-L6). At an angle of attack of 0 degrees, the transition is poorly resolved for mesh levels 1 to 3. At the higher angle of attack, the upper surface shows signs of inadequate mesh convergence. Families 2 and 4 showed similar behaviour to Families 1 and 3 respectively, hence are not shown here. Grid Family 5 shows good signs of mesh convergence, however, the results are analysed further in the section on the sensitivity of the number of spanwise elements. The unstructured meshes which were generated with the automated meshing workflow show much better grid convergence properties than the other grids. The only region that would benefit from further mesh refinement is the lower surface transition prediction for the case at 0 degrees alpha.

2D vs Pseudo-3D
^^^^^^^^^^^^^^^

Next 2D versus pseudo-3D predictions are analysed in greater detail. The skin friction distributions are compared between Family 1 and Family 2 as well as Family 3 and Family 4 at two mesh levels, shown in :ref:`Fig15_Cf2DvsPseudo3D`.

.. _Fig15_Cf2DvsPseudo3D:
.. figure:: Figures/CF_2DvsPseudo2D.png
   :align: center
   :width: 99%

   Comparison of the skin friction distribution for L2 and L6 grids at two angles of attack for 2D and pseudo-3D grids.

The skin friction distributions show that if the transition is underesolved as is the case for the L2 meshes, some impact of 2D vs pseudo-3D may be observed. However, once the skin friction gradient is resolved well by the mesh, the differences between 2D and pseudo-3D solutions become negligible. A minor sensitivity is seen on the lower surface at an angle of attack of 0, however, as will be seen in the analyses below, the transition location is very sensitive at this condition to the mesh.


Spanwise Grid Effects
^^^^^^^^^^^^^^^^^^^^^

The effect of number of spanwise elements is also analysed for meshes that contain both quad-type and tri-type elements on the surface. The primary reason for this investigation is that the integrated loads for grid Family 5 were an outlier compared to other grid families. As mentioned previously, an additional quad grid was generated with 40 spanwise nodes that does not belong to any grid family. To examine the effect of number of spanwise nodes, we first look at the surface skin friction and pressure distributions shown in :ref:`Fig16_CfSpan`.


.. _Fig16_CfSpan:
.. figure:: Figures/CPCF_Span.png
   :align: center
   :width: 99%

   Comparison of the skin friction and surface pressure distributions for L6 grids with varying number of spanwise nodes at two angles of attack.

For the quad grids, including Family 1, Family 2 and the grid with 40 nodes, very little sensitivity is seen in both the skin friction and surface pressure distributions, as the curves are pretty much on top of each other. A very minor sensitivity can be seen on the lower surface at 0 degrees angle of attack. The Family 6 grids show a very large sensitivity in the integrated loads, however, the skin friction distributions do not indicate such sensitivity. At both angles of attack the transition locations move slightly aft with mesh refinement, but not enough to cause larger differences in the integrated loads. Looking at the surface pressure distributions however, it can be seen that the tri grids with a low number of spanwise nodes, predict a larger acceleration over the leading edge of the airfoil as indicated by the larger suction peak. The trailing edge pressure recovery is also weaker. Based on these observations, it can be stated that the effect of number of spanwise nodes does not directly impact the transition location, but affects other critical regions of the flowfield to a greater degree. It must, however, be reiterated that the Family 6 grids with a lower number of spanwise nodes are of poorer grid quality due to the high aspect ratio's of the tet cells. Reducing the decay factor in the volume grid, does not change these observations either. To examine in further detail, exactly what impact the surface grid topology has on the transition location and three-dimensionality of the skin friction distribution, the skin friction contours are extracted at two angles of attack, shown in :ref:`Fig17_CfSpan_Top_0`-:ref:`Fig20_CfSpan_Bot_5`

.. _Fig17_CfSpan_Top_0:
.. figure:: Figures/CF_Span_TopSurf_0.png
   :align: center
   :width: 99%

   Comparison of the skin friction contours on the top surface for L6 grids with varying number of spanwise nodes at 0 degrees alpha.

.. _Fig18_CfSpan_Bot_0:
.. figure:: Figures/CF_Span_BotSurf_0.png
   :align: center
   :width: 99%

   Comparison of the skin friction contours on the lower surface for L6 grids with varying number of spanwise nodes at 0 degrees alpha.

.. _Fig19_CfSpan_Top_5:
.. figure:: Figures/CF_Span_TopSurf_5.png
   :align: center
   :width: 99%

   Comparison of the skin friction contours on the top surface for L6 grids with varying number of spanwise nodes at 5 degrees alpha.

.. _Fig20_CfSpan_Bot_5:
.. figure:: Figures/CF_Span_BotSurf_5.png
   :align: center
   :width: 99%

   Comparison of the skin friction contours on the lower surface for L6 grids with varying number of spanwise nodes at 5 degrees alpha.

The skin friction contours indicate that close to no three-dimensional features are seen in the quad grids. The tri surface grids, however, present a certain degree of three-dimensionality especially in the regions where the transition location was sensitive to grid refinement: the lower surface at 0 degrees angle of attack and the upper surface at 5 degrees angle of attack. At 0 degrees angle of attack, the level of three-dimensionality is fairly significant and irregular, whereas at 5 degrees angle of attack a minor waviness can be seen in the transition location which increases in frequency as the grid is refined. 

Grid Topology Effects
^^^^^^^^^^^^^^^^^^^^^
Analysing the effect of different grid elements further, the skin friction distributions are compared for the level 6 grids across different families :ref:`Fig21_CfElement`.

.. _Fig21_CfElement:
.. figure:: Figures/CF_ElementType.png
   :align: center
   :width: 99%

   Comparison of the skin friction distribution for L6 grids with different element types at two angles of attack.

Based on the skin friction distributions, it is confirmed that the regions of highest grid sensitivity in terms of transition location predictions are the lower surface at 0 degrees angle of attack and the upper surface at 5 degrees angle of attack. Even though fairly fine grids are used here, the transition location varies by over 0.1 chords at these conditions. This correlation, however, is likely to be improved if even finer grids were to be used. The differences are also due to the fact that quad and tri based meshes do not typically show the same integrated loads convergence with mesh refinement even for fully-turbulent solutions. The skin friction contours are extracted for grid Family 7 to assess what levels of three dimensionality is present at the transition locations, presented in :ref:`Fig22_VisCf_Uns`.

.. _Fig22_VisCf_Uns:
.. figure:: Figures/CF_TopBot_Unstructured.png
   :align: center
   :width: 99%

   Visualization of the skin friction contours for the L6 grids of Family 7 at two angles of attack.

The unstructured grids, also show a certain level of three-dimensionality in the skin friction contours, even though the span was reduced to 0.04 chords. Here, the largest effect can be seen on the lower surface at zero degrees angle of attack. 

3D Effects
^^^^^^^^^^

To examine whether the pseudo-3D results have similar behaviour as a real-life flow scenario, the transition location predictions are compared with a full 3D wing geometry. The 3D simulations were trimmed to the same lift coefficient. A particular focus is put on the level of three-dimensionality of the transition front, for a straight untapered untwisted wing. Firstly, we compare the skin friction coefficient distributions extract at mid-span for the 3D solution, presented in :ref:`Fig23_CfElement`.


.. _Fig23_CfElement:
.. figure:: Figures/CPCF_3D.png
   :align: center
   :width: 99%

   Comparison of the skin friction and surface pressure coefficient distribution between pseudo-3D and 3D solutions at two angles of attack.

The 3D geometry results show a high degree of consistency compared to the pseudo-3D unstructured grid results. Some minor differences can be seen especially in the resolution of the skin friction curve gradients for the 3D geometry. This is primarily due to the fact that a L6 grid resolution for a high-aspect ratio wing would lead to very large mesh sizes, hence a coarser mesh is used here. The transition locations are well captured and show good agreement. The surface pressure predictions, however, indicate a higher suction peak at both angles of attack for the 3D geometry. This means that the sectional lift coefficient is higher for the 3D geometry at the mid-span compared to the pseudo-3D case. The 3D sectional lift coefficient varies with span due to the presence of the trailed tip vortices, which is the cause for this difference.  Examining the results in further detail, the skin friction distributions are extracted for the 3D wing, shown in  :ref:`Fig24_VisCf_3D0deg`-:ref:`Fig25_VisCf_3D5deg`.

.. _Fig24_VisCf_3D0deg:
.. figure:: Figures/CF_3D_TopBot_0deg.png
   :align: center
   :width: 99%

   Visualization of the skin friction contours for the 3D grids at 0 degrees alpha.

.. _Fig25_VisCf_3D5deg:
.. figure:: Figures/CF_3D_TopBot_5deg.png
   :align: center
   :width: 99%

   Visualization of the skin friction contours for the 3D grids at 5 degrees alpha.

The skin friction distributions show limited three-dimensional effects. At 0 degrees angle of attack some minor irregularity can be seen in the mid-span region of the wing. However, at this condition, the pressure gradients are not significant, hence numerical and discretization sensitivities are emphasized. The 5 degree angle of attack case shows a much more regular transition front. The magnitude of the skin friction peak reduces towards the tip of the wing at both angles of attack. This leads to a spanwise variation in the transition location at 5 degrees angle of attack. Based on the 3D vs pseudo-3D results, it can be stated that the transition locations have good correlation, however, to examine the exact structure of the transition front in terms of three-dimensional effects, finer grids are required.

Comparisons with FUN3D/OVERFLOW and Experimental Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Next we move onto comparisons of the transition predictions using the Flow360 solver with other CFD codes and experimental data. Firstly we compare the integrated loads convergence with mesh refinement with `data from OVERFLOW <https://arc.aiaa.org/doi/10.2514/6.2019-0039>`_ using the SA-AFT transition model and `OVERFLOW/FUN3D <https://arc.aiaa.org/doi/abs/10.2514/6.2022-3679>`_ using the :math:`\gamma- Re_\theta` transition model. The mesh convergence study comparisons are presented in :ref:`Fig26_CL0Exp`-:ref:`Fig27_CD0Exp2`.

.. _Fig26_CL0Exp:
.. figure:: Figures/CL0_Exp.png
   :align: center
   :width: 99%

   Convergence of the lift coefficient with mesh refinement at :math:`\alpha = 0^o` for different CFD codes and transition models.

.. _Fig26_CL0Exp2:
.. figure:: Figures/CL5_Exp.png
   :align: center
   :width: 99%

   Convergence of the lift coefficient with mesh refinement at :math:`\alpha = 5^o` for different CFD codes and transition models.

.. _Fig27_CD0Exp:
.. figure:: Figures/CD0_Exp.png
   :align: center
   :width: 99%

   Convergence of the drag coefficient with mesh refinement at :math:`\alpha = 0^o` for different CFD codes and transition models.

.. _Fig27_CD0Exp2:
.. figure:: Figures/CD5_Exp.png
   :align: center
   :width: 99%

   Convergence of the drag coefficient with mesh refinement at :math:`\alpha = 5^o` for different CFD codes and transition models.

The Flow360 results show similar convergence histories with mesh refinement as FUN3D and OVERFLOW. At 0 degrees angle of attack, a nonlinearity is present going from the L3 to the L4 grid, due to a shift in the transition location on the upper surface. The lift and drag coefficient predictions trend to the same values as other CFD codes, although finer meshes are required to attain mesh convergence, similarly as examined in the :math:`\gamma- Re_\theta` results. Differences can also be seen in the values obtained by the two different transition models. The slopes of the integrated loads convergence curves with mesh refinement, indicate that OVERFLOW is less sensitive to mesh resolution than Flow360 or FUN3D. This is due to the fact that OVERFLOW is a 3rd order cell-centred structured solver, whereas FUN3D and Flow360 are node centred unstructured 2nd order solvers. This means that both FUN3D and Flow360 require a finer grid than OVERFLOW to attain similar levels of accuracy. On the other hand, the primary benefit of unstructured solvers is easier control of the mesh refinement regions and more efficient use of mesh adaptation. The use of mesh adaptation was demonstrated for the NLF(1)-0416 airfoil by `Venkatachari et al. <https://arc.aiaa.org/doi/abs/10.2514/6.2022-3679>`_ who obtained similar predictions as OVERFLOW using adapted meshes in FUN3D for a given grid size. Before performing further comparisons with experimental data and other CFD codes, a finer 2D quad-based mesh was generated with 2.168 million nodes. The skin friction and surface pressure distributions are compared between the different CFD solvers and transition models in :ref:`Fig28_CfExp`. Note that here, the OVERFLOW and FUN3D predictions based on the :math:`\gamma- Re_\theta` model are presented for L8 committee grids (unavailable in the public domain), whereas the OVERFLOW SA-AFT results are presented for the L3 committee grid.

.. _Fig28_CfExp:
.. figure:: Figures/CpCf_Exp.png
   :align: center
   :width: 99%

   Comparison of the skin friction and surface pressure coefficient distribution between different CFD codes and transition models.

The Flow360 skin friction and surface pressure distributions show good agreement with the OVERFLOW and FUN3D solutions. At 0 degrees angle of attack, the transition locations are in excellent agreement with minor deviations in the skin friction coefficient peak values. At 5 degrees angle of attack, the differences are slightly greater, however different transition models and grids were used here. The surface pressure predictions are in good agreement, with similar differences from experimental data observed in all CFD solvers and transition models. Finally, a full alpha sweep is performed and compared with experimental data and the SA-AFT OVERFLOW results (for the L3 committee grid), shown in :ref:`Fig29_Polars`-:ref:`Fig29_Polars2`. Also included are fully-turbulent SA results for both solvers.

.. _Fig29_Polars:
.. figure:: Figures/Polars.png
   :align: center
   :width: 99%

   Comparison of the CL vs alpha curves between Flow360 and OVERFLOW for the SA and SA-AFT turbulence models.

.. _Fig29_Polars2:
.. figure:: Figures/Polars2.png
   :align: center
   :width: 99%

   Comparison of the drag polar curves between Flow360 and OVERFLOW for the SA and SA-AFT turbulence models.

The alpha sweep results also show good correlation of Flow360 results with OVERFLOW and experimental data. The lift coefficient is slightly overpredicted compared to experiments at a given angle of attack, with excellent agreement with OVERFLOW predictions for both the SA and SA-AFT models. The Flow360 drag polar also shows very good agreement with both experimental data and OVERFLOW results, although the curve is underesolved due to simulations at 2 degree increments in alpha. Both Flow360 and OVERFLOW predict a slightly narrower drag bucket than experiments. The SA-AFT drag polar shows the importance of including transition modeling for the present case, as the drag is overpredicted when using the SA model. Furthermore, DES-based (Detached Eddy Simulation) turbulence modeling is recommended for simulations past stall, as the RANS-based solutions overpredict the lift coefficient at high angles of attack for both Flow360 and OVERFLOW cases. 


Conclusions
-----------

This validation study leads to the following recommendations and conclusions for investigations that include transition modelling:

- Transition location predictions are sensitive to grid resolution. Grid independent results require meshes in the millions of nodes for 2D/pseudo-3D configurations, meaning that for 3D geometries very fine meshes are required. Coarser meshes are able to predict transition with a minor error on the transition location, especially on the airfoil suction side at positive angles of attack and airfoil pressure side at negative angles of attack.

- Quad and Tri based meshes approach the convergence limit from different directions, hence to gain full confidence in the result, a mesh refinement study is always recommended for each configuration.

- Quad-dominant surface meshes with quads oriented in the primary flow direction are preferred to tri-dominant meshes for configurations which can be treated as close to pseudo-3D. The primary reason for this is that tri-dominant surface meshes introduce three-dimensionality due to the discretization even for purely 2D configurations. 

- The effect of pseudo-3D vs 2D is negligible for quad-dominant surface meshes. For tri-dominant surface meshes, the effect on the integrated loads is considerable especially when the cell aspect ratio's are large in the spanwise direction. To minimize the impact of quad vs tri meshes for three-dimensional configurations, the cell aspect ratio's should be kept below 100 which is the case even for non-transitional calculations. The cell aspect ratio's are less important for quad-dominant grids.

- Tri-based meshes are generally less sensitive to mesh refinement than quad based meshes although lead to much higher dissipation for coarser meshes in terms of transition location predictions and the gradient resolution of the skin friction curves.

- The Flow360 solver was validated against experimental data and other CFD codes, showing good correlation in the integrated loads and transition location predictions for both the flat plate and NLF(1)-0416 airfoil cases.

