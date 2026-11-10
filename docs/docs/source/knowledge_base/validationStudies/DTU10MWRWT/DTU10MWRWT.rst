.. _windTurbineValidationStudy:

.. |agr|     unicode:: U+003B1 .. GREEK SMALL LETTER ALPHA
.. |omega|   unicode:: U+03C9  .. OMEGA SIGN

DTU 10MW Wind Turbine
===============================

Introduction
------------

The purpose of this validation study is to validate and demonstrate the application of Flow360 to wind turbine simulations. 
A common research wind turbine model, `the DTU 10MW Reference Wind Turbine (RWT) <https://orbit.dtu.dk/en/publications/the-dtu-10-mw-reference-wind-turbine>`_, 
was selected and unsteady compressible DDES simulations performed with the Spalart-Allmaras (SA) turbulence model. 
Flow conditions were simulated at the rated wind speed of 11m/s, which relate to the rated power output of 10MW. 
Off-design conditions with different wind speeds were also simulated.

Rotor Geometry
--------------

The wind turbine blade geometry was reproduced programmatically using 200 spanwise cross-sections along the blade and airfoil profiles extracted from the original source geometry (see :ref:`geoSchematics_windTurbineValidationStudy`). 
The fine resolution blade geometry from `the GitLab repository of DTU 10MW RWT <https://gitlab.windenergy.dtu.dk/rwts/dtu-10mw-rwt>`_ was used.

.. _geoSchematics_windTurbineValidationStudy:
.. figure:: Figures/GeoSchematics.png
   :align: center

   Schematic of sectional airfoil profiles extracted. For clarity, only 100 cross-sections are shown.

The original blade geometry used is rigid with no aeroelastic deformations. 
Slight modifications were made at the blade tip to help improve unstructured mesh quality. 
Moreover, an elliptic fairing was added to reduce the downstream separation region around the blade roots. 
:ref:`geoErrorCheck_windTurbineValidationStudy` displays an overlay of the geometry created here and the original. 
The current CAD model is an EGADS format and is available `here <https://simcloud-public-1.s3.amazonaws.com/dtu10MWrwt/geometry.egads>`_.

.. _geoErrorCheck_windTurbineValidationStudy:
.. figure:: Figures/GeoErrorCheck.png
   :align: center

   Geometric comparison of wind turbine models (Red: original design, Green: current model).

Automated Meshing
-----------------

A series of computational grids, with consecutively refined mesh resolution, were created with our automated meshing process. The corresponding key meshing parameters are summarized in :ref:`meshStatistics_windTurbineValidationStudy`. The volume mesh was consistently refined by a factor of :math:`2`. Additionally, the edge lengths orthogonal to curves, such as the leading edge (LE), the trailing edge (TE) and the blade tip, were refined by a factor of :math:`2^{1/3}`. A refinement factor of :math:`2^{2/3}` was also applied to the max cell edge length on the blade and the tip faces. Notably, the target 1st layer thicknesses were kept as 1E-06 meters, but are influenced by the volumetric refinement factor. The 1st layer thickness values applied were meant to ensure :math:`y^+ < 1` in all the cases. The farfield was automatically set to be about 7700 meters away from the rotor center.      

.. _meshStatistics_windTurbineValidationStudy:
.. csv-table:: Mesh statistics 
   :file: Tables/meshStatistics.csv
   :widths: 20, 6, 8, 8, 8
   :align: center
   :header-rows: 1
   :delim: @

The medium mesh resolution is visualized in :ref:`standardMesh_windTurbineValidationStudy`. 
The upper two frames display the rotor nearfield and a zoomed-in view, respectively. 
As seen in the upper left figure, a rotating mesh region surrounding the rotor was used with sliding interfaces and a maximum element edge length of 1 meter applied. 
To better resolve the downstream progression of tip vortices, an additional volumetric refinement region was introduced with a 2 meter cell spacing. 
The front and rear views of the surface mesh are shown in the lower left and lower right frames, respectively. 
Refinement of the surface mesh can be seen along the LE and TE, as well as at the blade root and tip.

.. _standardMesh_windTurbineValidationStudy:
.. figure:: Figures/Mesh-StandardGridDensity.png
   :figwidth: 99%
   :align: center

   Overview of surface and volume mesh (medium resolution).

The configuration json files are available through these links for the aforementioned surface (`Coarse <https://simcloud-public-1.s3.amazonaws.com/dtu10MWrwt/surfaceMesh_Coarse.json>`__, `Medium <https://simcloud-public-1.s3.amazonaws.com/dtu10MWrwt/surfaceMesh_Standard.json>`__ and `Fine <https://simcloud-public-1.s3.amazonaws.com/dtu10MWrwt/surfaceMesh_Fine.json>`__) and volume (`Coarse <https://simcloud-public-1.s3.amazonaws.com/dtu10MWrwt/volumeMesh_Coarse.json>`__, `Medium <https://simcloud-public-1.s3.amazonaws.com/dtu10MWrwt/volumeMesh_Standard.json>`__ and `Fine <https://simcloud-public-1.s3.amazonaws.com/dtu10MWrwt/volumeMesh_Fine.json>`__) meshes, respectively.

Flow Configuration
------------------

The wind turbine operation was first simulated at the rated power condition, utilizing the three computational grids discussed above.
Key flow configuration parameters are summarized in :ref:`solverConfig_windTurbineValidationStudy`. 
Importantly, to extract power from the air, the rotor is designed for incoming wind to impinge on the pressure side of the turbine blades, 
so that the circumferential component of the aerodynamic lift drives the turbine rotation in the positive z-direction (right-hand rule). 
This is equivalent to setting the angle of attack, :math:`\alpha = +90` |deg|, 
in accordance with :ref:`Flow360's conventions<alphaBetaAngles>`. 
Additional details for solver settings are given in the :ref:`Navier-Stokes <knowledge_base_navierStokesSolver>` and :ref:`turbulence model<knowledge_base_turbulenceModelSolver>` knowledge bases.

.. _solverConfig_windTurbineValidationStudy:
.. csv-table:: Key flow configuration parameters
   :file: Tables/solverConfig.csv
   :widths: 20, 10, 10, 20
   :align: center
   :header-rows: 1
   :delim: @

Unsteady DDES simulations were carried out through three consecutive stages. To initialize the flow field, the turbine operation was started with a physical time step size equivalent to rotating through :math:`6` |deg| for three revolutions, utilizing 1st-order spatial schemes for both Navier-Stokes and turbulence model equations. This was followed by twelve revolutions with 2nd-order schemes through the second stage to further develop the flow. Finally, for time-accurate solutions, the physical time step size was reduced to :math:`3` |deg|, :math:`2` |deg| and :math:`1` |deg| per time step, according to the three grid levels, and simulations were carried on for another three revolutions. The solver configuration json files are available here (`Stage-1 <https://simcloud-public-1.s3.amazonaws.com/dtu10MWrwt/Flow360_Stage1_1stOrd_6Deg.json>`_, `Stage-2 <https://simcloud-public-1.s3.amazonaws.com/dtu10MWrwt/Flow360_Stage2_2ndOrd_6Deg.json>`_, and Stage-3 with `3DegPerStep <https://simcloud-public-1.s3.amazonaws.com/dtu10MWrwt/Flow360_Stage3_2ndOrd_3Deg_Upd3.json>`_, `2DegPerStep <https://simcloud-public-1.s3.amazonaws.com/dtu10MWrwt/Flow360_Stage3_2ndOrd_2Deg_Upd3.json>`_ and `1DegPerStep <https://simcloud-public-1.s3.amazonaws.com/dtu10MWrwt/Flow360_Stage3_2ndOrd_1Deg_Upd3.json>`_). More details about time integration settings are given in the :ref:`time stepping knowledge base <knowledge_base_timeStepping>`. The corresponding temporal evolutions of CFL numbers, nonlinear residuals, force and moment coefficients are shown in the next section for the case with the medium grid.
   
Residual Convergence
---------------------

Within each physical time step, the CFL number is ramped up from 10 to 200 in 10 pseudo steps. The maximum number of pseudo steps within each physical time step is conservatively set to 100 to allow all nonlinear residuals to decrease by at least two orders of magnitude. Convergence of nonlinear residuals is shown in :ref:`nonlinearResiduals_windTurbineValidationStudy`. For clarity, only continuity is displayed.

.. _nonlinearResiduals_windTurbineValidationStudy:
.. figure:: Figures/Log_NonlinearResiduals.png
   :align: center

   Convergence of continuity nonlinear residual, medium grid, stage 3.
   
The temporal evolutions of force and moment coefficients are shown in :ref:`CFxyz_windTurbineValidationStudy` and :ref:`CMxyz_windTurbineValidationStudy`. As seen, the aerodynamic loads seemed to be stabilized, with minor blade passing oscillations.

.. _CFxyz_windTurbineValidationStudy:
.. figure:: Figures/Log_CFxyz.png
   :align: center

   Evolutions of force coefficients (CFx, CFy and CFz), medium grid, stage 3.
   
.. _CMxyz_windTurbineValidationStudy:
.. figure:: Figures/Log_CMxyz.png
   :align: center

   Evolutions of moment coefficients (CMx, CMy and CMz), medium grid, stage 3.


Surface and Volumetric Results
------------------------------

Numerical results computed on the mesh of the medium resolution are examined in this section. Surface distributions of :math:`y^{+}`, :math:`C_{p}` and :math:`C_{f}`, are displayed on one of the three blades, respectively. The elliptic fairing is contoured by :math:`C_{p}`. As seen, all values of :math:`y^{+}` were less than :math:`1`. Also, as expected, the major fraction of power output was generated from the outer part of the span primarily by the pressure differences between the pressure and the suction sides.  
   
   
.. _surfaceResults_windTurbineValidationStudy:
.. figure:: Figures/surfaceResults.png
   :figwidth: 99%
   :align: center
   
   Surface contours of :math:`y^{+}`, :math:`C_{p}` and :math:`C_{f}`.

Tip vortices were identified by Q-criterion and visualized as shown in the next figure, which were contoured by Mach number. Root vortices separating off the fairing, as well as the Gurney flaps, were also identified.
   
.. _volumeResults_windTurbineValidationStudy:
.. figure::  Figures/volumeResults.png
   :figwidth: 99%
   :align: center   
   
   Iso-surfaces of Q-criterion contoured by Mach number.


Sectional Loads on One of the Blades
------------------------------------

To validate Flow360 results, sectional loads are examined in this section, with respect to `the DTU CFD data <https://gitlab.windenergy.dtu.dk/rwts/dtu-10mw-rwt>`_. These sectional data were extracted from the flow solutions associated with one of the blades, that aligned with the positive direction of y-axis, for example, as shown in the above two figures. Following Flow360's definitions, the forces shown in the following figures, and so on, are imposed on the turbine rotor by the fluid. 

Spanwise distributions of the circumferential component of aerodynamic force are compared in :ref:`sectionalLoads_forceX_windTurbineValidationStudy`. As expected, it was this force component that drove the rotor spinning around the positive z-direction. This is consistent with the :math:`C_{p}` distributions shown above.

.. _sectionalLoads_forceX_windTurbineValidationStudy:
.. figure:: Figures/SectionalLoads_Fig1_ForceX.png
   :align: center

   Tangential force as function of radius.
   
Spanwise distributions of the axial component of aerodynamic force are compared in :ref:`sectionalLoads_forceZ_windTurbineValidationStudy`. Some discrepancies are present between Flow360 results and the reference at the outer blade, spanning from about 55 to 85 meters. In addition to the aforementioned differences in the turbine geometries, the differences between the current and reference simulation methodologies are likely contributing here, as summarized in :ref:`solverFeatures_windTurbineValidationStudy`. DTU results were computed on a structured grid with 14.1 million cells utilizing a pressure-based incompressible Reynolds-averaged Navier-Stokes (RANS) solver, where steady state simulations were performed with rotating effects analytically described as discussed, for example, by `Troldborg et al <https://iopscience.iop.org/article/10.1088/1742-6596/753/2/022057/pdf>`_ and `Johansen et al <https://backend.orbit.dtu.dk/ws/portalfiles/portal/283945141/Numerical_Investigation_of_a_Wind_Turbine_Rotor_wi.pdf>`_. Whereas, unsteady compressible DDES simulations were performed with Flow360, where a rotating mesh region was incorporated with sliding interfaces.  It is suspected that weak compressibility effects may also contribute to such a discrepancy.
   
.. _sectionalLoads_forceZ_windTurbineValidationStudy:
.. figure:: Figures/SectionalLoads_Fig2_ForceZ.png
   :align: center

   Axial force as function of radius.
   
.. _solverFeatures_windTurbineValidationStudy:
.. csv-table:: Key flow solver features 
   :file: Tables/solverFeatures.csv
   :widths: 15, 15, 15
   :align: center
   :header-rows: 1
   :delim: @
   
Local thrust coefficient, as defined by :math:numref:`LocalThrustCoef`, are compared in :ref:`sectionalLoads_thrustCoef_windTurbineValidationStudy`. In the right hand side of :math:numref:`LocalThrustCoef`, all quantities are dimensional. Here and in the next figure, a constant value of 3.4 meters was used for the reference chord length :math:`L_{chord,ref}`. Also, :math:`F_z(r)` is the sectional axial force. As seen from the figure, there are relatively large differences near the blade root. This likely can be attributed to the presence of the elliptic fairing, together with grid density differences. However, these effects are marginal to the integrated power output.

.. math::
      :label: LocalThrustCoef
      
      C_{thrust}(r) = \frac{F_z(r)}{\frac{1}{2} \rho_{\infty}(\omega \cdot r)^2 \cdot L_{chord,ref}} \cdot \frac{r}{R_{tip}}
   
.. _sectionalLoads_thrustCoef_windTurbineValidationStudy:
.. figure:: Figures/SectionalLoads_Fig3_LocalCt.png
   :align: center

   Local thrust coefficient as function of radius.
   
Local power coefficients, as defined by :math:numref:`LocalPowerCoef`, are compared in :ref:`sectionalLoads_powerCoef_windTurbineValidationStudy`. Notably, in the definition, :math:`M_z(r)` is the sectional axial moment. As seen from the figure, results agree well between current simulations on the three meshes and that of DTU CFD data, especially for spans greater than 10 meters. Contributions to the power output were decomposed into three segments along the blade, the inner (4 to 33 meters), the middle (33 to 61 meters) and the outer (61 to 89 meters) parts. The corresponding fractions to the total power were about 13%, 38% and 49%. 

.. math::
      :label: LocalPowerCoef
      
      C_{power}(r) = \frac{M_z(r)}{\frac{1}{2} \rho_{\infty}(\omega \cdot r)^2 \cdot L_{chord,ref}} \cdot \frac{r}{R_{tip}} \cdot \frac{\omega}{U_{\infty}}
   
.. _sectionalLoads_powerCoef_windTurbineValidationStudy:
.. figure:: Figures/SectionalLoads_Fig4_LocalCp.png
   :align: center

   Local power coefficient as function of radius.
  

Integrated Loads on the Entire Rotor
------------------------------------

The integrated data of axial force and moment, as well as power output are summarized in :ref:`integratedLoads_windTurbineValidationStudy`. Flow360 results were averaged through the last one revolution. DTU results are also given in this table as `references <https://gitlab.windenergy.dtu.dk/rwts/dtu-10mw-rwt>`_. Overall, reasonable agreement was achieved, with the maximum relative errors of 6.2% and 3.7% for integrated values of axial force and power, respectively.    

.. _integratedLoads_windTurbineValidationStudy:
.. csv-table:: Integral axial force and moment coefficients, their dimensional values and the integrated power 
   :file: Tables/integratedLoads.csv
   :widths: 6, 6, 10, 10, 10, 8
   :align: center
   :header-rows: 1
   :delim: @

For Flow360 results, it seems that these integrated performance parameters were asymptotically converging along with the grid refinements, as shown in :ref:`gridConvergence_windTurbineValidationStudy`. For the results computed on the coarse mesh, the relative errors of these dimensional quantities are all within 1.5%, with respect to those on the fine mesh. To reserve computational costs, off-design simulations were performed using the coarse mesh, as discussed in the next section with more details.

.. _gridConvergence_windTurbineValidationStudy:
.. figure:: Figures/gridConvergence.png
   :align: center

   Grid convergence. The lower x-axis is the number of grid nodes labeled as :math:`N`, and the corresponding upper axis shows the scale of the expected numerical error labeled as :math:`N^{-2/3}`. Dimensional values of axial force and mechanical power are shown in the upper and the lower frames, respectively. 
   
Off-Design Performance
----------------------

For validation and demonstration purposes, wind turbine operations were further simulated at off-design conditions. To limit engineering efforts without having to further reproduce more turbine geometries, four more off-design conditions with the same zero blade pitch angle as that operated at the rated condition were selected. The corresponding freestream wind speeds were 7.0, 8.0, 9.0 and 10.0 m/s. Also, as discussed above, the coarse mesh was used in these simulations.

Off-design sectional loads on one of the blades are shown in :ref:`offDesign_sectionalLoads_forceX_windTurbineValidationStudy`, :ref:`offDesign_sectionalLoads_forceZ_windTurbineValidationStudy`, :ref:`offDesign_sectionalLoads_thrustCoef_windTurbineValidationStudy` and :ref:`offDesign_sectionalLoads_powerCoef_windTurbineValidationStudy` for the spanwise distributions of tangential and axial forces, and local thrust and local power coefficients, respectively. For comparison, those of the rated power condition are also shown in these figures, together with the available DTU CFD data. As seen from these figures, overall agreement was well achieved. Notably, Flow360 results at the wind speed of 7.0 m/s are not shown in these figures for sectional loads, since the counterparts of DTU CFD data were not available.

.. _offDesign_sectionalLoads_forceX_windTurbineValidationStudy:
.. figure:: Figures/OffDesign_SectionalLoads_Fig1_ForceX.png
   :align: center

   Tangential force as function of radius.
   

.. _offDesign_sectionalLoads_forceZ_windTurbineValidationStudy:
.. figure:: Figures/OffDesign_SectionalLoads_Fig2_ForceZ.png
   :align: center

   Axial force as function of radius.
   
   
.. _offDesign_sectionalLoads_thrustCoef_windTurbineValidationStudy:
.. figure:: Figures/OffDesign_SectionalLoads_Fig3_LocalCt.png
   :align: center

   Local thrust coefficient as function of radius.
   
   
.. _offDesign_sectionalLoads_powerCoef_windTurbineValidationStudy:
.. figure:: Figures/OffDesign_SectionalLoads_Fig4_LocalCp.png
   :align: center

   Local power coefficient as function of radius.
   
The off-design key performance parameters, including integrated rotor power and thrust, and their coefficients, are summarized in :ref:`offDesignPerformance_windTurbineValidationStudy`. The definitions of these integral power and thrust coefficients are given in :math:numref:`IntegralPowerCoef` and :math:numref:`IntegralThrustCoef`. Notably, Flow360 results were averaged through the last one revolution.

.. math::
      :label: IntegralPowerCoef
      
      C_P = \frac{Power}{\frac{1}{2} \rho_{\infty} (\pi \cdot R_{tip}^2) U_{\infty}^3}

.. math::
      :label: IntegralThrustCoef
      
      C_T = \frac{Thrust}{\frac{1}{2} \rho_{\infty} (\pi \cdot R_{tip}^2) U_{\infty}^2}
      
.. _offDesignPerformance_windTurbineValidationStudy:
.. csv-table:: Integrated rotor power and thrust and their coefficients as a function of wind speed. 
   :file: Tables/offDesignPerformance.csv
   :widths: 20, 6, 6, 6, 6, 6
   :align: center
   :header-rows: 1
   :delim: @
   
Off-design characteristics are visualized in :ref:`offDesignCharacteristics_windTurbineValidationStudy`, where key performance parameters as summarized in the above table are shown in each frame with DTU CFD data. Flow360 predictions reasonably captured the simulated off-design performance . Off-design operations above the rated wind speed, 11m/s, can be expected to produce large flow separation from rotor blades. For these scenarios, DDES simulations would provide improved predictions with respect to RANS simulations. 
   
.. _offDesignCharacteristics_windTurbineValidationStudy:

.. figure:: Figures/offDesignPerformance.png
   :align: center
   
   Power and thrust and their coefficients as function of wind speed.
