.. _python_api_example_library:

.. currentmodule:: flow360

*****************
Examples Library
*****************

.. raw:: html

   <div class="ex-filterbar">
      <!-- Time -->
      <div class="ex-group" data-group="time">
         <span class="ex-group-label">Time</span>
         <button class="ex-filter is-active" data-filter="all" data-group="time">All</button>
         <button class="ex-filter" data-filter="steady"   data-group="time">Steady</button>
         <button class="ex-filter" data-filter="unsteady" data-group="time">Unsteady</button>
      </div>

      <!-- Turbulence / Modeling -->
      <div class="ex-group" data-group="turbulence">
         <span class="ex-group-label">Turbulence Modeling</span>
         <button class="ex-filter is-active" data-filter="all" data-group="turbulence">All</button>
         <button class="ex-filter" data-filter="rans"       data-group="turbulence">RANS</button>
         <button class="ex-filter" data-filter="ddes"       data-group="turbulence">DDES</button>
         <button class="ex-filter" data-filter="laminar"    data-group="turbulence">Laminar</button>
         <button class="ex-filter" data-filter="transition" data-group="turbulence">Transition</button>
      </div>

      <!-- Models / Features -->
      <div class="ex-group" data-group="features">
         <span class="ex-group-label">Features</span>
         <button class="ex-filter is-active" data-filter="all" data-group="features">All</button>
         <button class="ex-filter" data-filter="aeroacoustics" data-group="features">Aeroacoustics</button>
         <button class="ex-filter" data-filter="actuator-disk" data-group="features">Actuator Disk</button>
         <button class="ex-filter" data-filter="bet"          data-group="features">BET</button>
         <button class="ex-filter" data-filter="mrf"          data-group="features">MRF</button>
         <button class="ex-filter" data-filter="cht"          data-group="features">CHT</button>
         <button class="ex-filter" data-filter="periodic"     data-group="features">Periodic</button>
         <button class="ex-filter" data-filter="roughness"    data-group="features">Roughness</button>
         <button class="ex-filter" data-filter="udd"          data-group="features">UDD</button>
         <button class="ex-filter" data-filter="stopping-criteria"          data-group="features">Custom Stopping Criteria</button>
      </div>

      <!-- Geometry / Application -->
      <div class="ex-group" data-group="geometry">
         <span class="ex-group-label">Application</span>
         <button class="ex-filter is-active" data-filter="all" data-group="geometry">All</button>
         <button class="ex-filter" data-filter="aerospace"        data-group="geometry">Aerospace</button>
         <button class="ex-filter" data-filter="rotorcraft"           data-group="geometry">Rotorcraft</button>
         <button class="ex-filter" data-filter="automotive"     data-group="geometry">Automotive</button>
         <button class="ex-filter" data-filter="liquid"         data-group="geometry">Liquid</button>
         <button class="ex-filter" data-filter="turbomachinery" data-group="geometry">Turbomachinery</button>
      </div>

      <!-- Meshing Workflow -->
      <div class="ex-group" data-group="meshing_workflow">
         <span class="ex-group-label">Meshing Workflow</span>
         <button class="ex-filter is-active" data-filter="all" data-group="meshing_workflow">All</button>
         <button class="ex-filter" data-filter="legacy"          data-group="meshing_workflow">Legacy</button>
         <button class="ex-filter" data-filter="new"         data-group="meshing_workflow">New</button>
         <button class="ex-filter" data-filter="geometry_ai"         data-group="meshing_workflow">GeometryAI</button>
         <button class="ex-filter" data-filter="snappy"         data-group="meshing_workflow">Snappy</button>
      </div>

      <!-- Starting Point -->
      <div class="ex-group" data-group="starting_point">
         <span class="ex-group-label">Starting Point</span>
         <button class="ex-filter is-active" data-filter="all" data-group="starting_point">All</button>
         <button class="ex-filter" data-filter="from_geometry"          data-group="starting_point">From geometry</button>
         <button class="ex-filter" data-filter="from_surface_mesh"         data-group="starting_point">From surface mesh</button>
         <button class="ex-filter" data-filter="from_volume_mesh"         data-group="starting_point">From volume mesh</button>
      </div>

      <!-- Others -->
      <div class="ex-group" data-group="others">
         <span class="ex-group-label">Others</span>
         <button class="ex-filter is-active" data-filter="all" data-group="others">All</button>
         <button class="ex-filter" data-filter="sweep"          data-group="others">Sweep</button>
         <button class="ex-filter" data-filter="report"         data-group="others">Report</button>
      </div>
   </div>

.. grid:: 3
   :gutter: 2
   :class-container: ex-grid

   .. grid-item-card:: Aeroacoustics
      :link: notebooks/aeroacoustics
      :link-type: doc
      :class-card: tag-unsteady tag-rans tag-aeroacoustics tag-rotorcraft tag-aerospace tag-from_volume_mesh
      
      .. image:: thumbnails/aeroacoustics.png
         :class: gallery-img

   .. grid-item-card:: Cold-Started Alpha Sweep
      :link: notebooks/alpha_sweep
      :link-type: doc
      :class-card: tag-steady tag-rans tag-sweep tag-report tag-aerospace tag-from_geometry tag-legacy
      
      .. image:: thumbnails/alpha_sweep.png
         :class: gallery-img

   .. grid-item-card:: Conjugate Heat Transfer
      :link: notebooks/conjugate_heat_transfer
      :link-type: doc
      :class-card: tag-steady tag-rans tag-cht tag-report tag-from_volume_mesh
      
      .. image:: thumbnails/conjugate_heat_transfer.png
         :class: gallery-img

   .. grid-item-card:: DARPA SUBOFF Actuator Disk
      :link: notebooks/DARPA_SUBOFF_AD
      :link-type: doc
      :class-card: tag-steady tag-rans tag-actuator-disk tag-liquid tag-from_geometry tag-new
      
      .. image:: thumbnails/darpa_geo.png
         :class: gallery-img

   .. grid-item-card:: Dynamic Derivatives Using Sliding Interfaces
      :link: notebooks/dynamic_derivatives
      :link-type: doc
      :class-card: tag-unsteady tag-rans tag-aerospace tag-from_geometry tag-legacy

      .. image:: thumbnails/dynamic_derivatives.png
         :class: gallery-img

   .. grid-item-card:: F1 Car Demo
      :link: notebooks/F1_car_demo
      :link-type: doc
      :class-card: tag-steady tag-rans tag-automotive tag-from_geometry tag-snappy

      .. image:: thumbnails/F1_car_demo.png
         :class: gallery-img

   .. grid-item-card:: Flat Plate with Structural Aerodynamic Load
      :link: notebooks/flat_plate_spring
      :link-type: doc   
      :class-card: tag-unsteady tag-rans tag-udd tag-from_volume_mesh

      .. image:: thumbnails/flat_plate_qcriterion.png
         :class: gallery-img

   .. grid-item-card:: Hinge Torques Monitoring
      :link: notebooks/hinge_torques
      :link-type: doc
      :class-card: tag-steady tag-rans tag-aerospace tag-report tag-from_geometry tag-legacy
      
      .. image:: thumbnails/aileron_rudder_airplane_geometry.png
         :class: gallery-img

   .. grid-item-card:: Isolated Wing with Propeller BET Model
      :link: notebooks/isolated_wing_BET
      :link-type: doc
      :class-card: tag-steady tag-rans tag-bet tag-aerospace tag-rotorcraft tag-from_geometry tag-new
      
      .. image:: thumbnails/isolated_wing_BET.png
         :class: gallery-img

   .. grid-item-card:: MRF Rotor
      :link: notebooks/MRF_rotor
      :link-type: doc
      :class-card: tag-steady tag-rans tag-mrf tag-rotorcraft tag-aerospace tag-from_geometry tag-new
      
      .. image:: thumbnails/poster_image.png
         :class: gallery-img

   .. grid-item-card:: Periodic BCs
      :link: notebooks/periodic_bc
      :link-type: doc
      :class-card: tag-steady tag-rans tag-periodic tag-turbomachinery tag-from_volume_mesh
      
      .. image:: thumbnails/periodic_turbo.png
         :class: gallery-img

   .. grid-item-card:: RANS CFD on 2D CRM Airfoil
      :link: notebooks/2D_CRM_airfoil
      :link-type: doc
      :class-card: tag-steady tag-rans tag-aerospace tag-legacy tag-from_geometry

      .. image:: thumbnails/2D_CRM_airfoil.png
         :class: gallery-img

   .. grid-item-card:: SRF with Cube
      :link: notebooks/cube_snappy
      :link-type: doc
      :class-card: tag-steady tag-rans tag-snappy tag-mrf tag-from_geometry
      
      .. image:: thumbnails/cube_snappy.png
         :class: gallery-img

   .. grid-item-card:: Time-Accurate BET eVTOL Simulation
      :link: notebooks/BET_eVTOL
      :link-type: doc
      :class-card: tag-unsteady tag-ddes tag-bet tag-rotorcraft tag-aerospace tag-from_geometry tag-legacy

      .. image:: thumbnails/bet_evtol.png
         :class: gallery-img

   .. grid-item-card:: Time-Accurate Rotor with Thrust-Convergence Stopping Criterion
      :link: notebooks/time_accurate_wind_turbine
      :link-type: doc
      :class-card: tag-unsteady tag-rans tag-rotorcraft tag-aerospace tag-from_geometry tag-new
      
      .. image:: thumbnails/wind_turbine.png
         :class: gallery-img

   .. grid-item-card:: Time-Accurate XV-15 Rotor
      :link: notebooks/time_accurate_XV_15
      :link-type: doc
      :class-card: tag-unsteady tag-ddes tag-rotorcraft tag-aerospace tag-from_volume_mesh
      
      .. image:: thumbnails/time_accurate_XV_15.png
         :class: gallery-img

   .. grid-item-card:: Transition Model 2D Airfoil
      :link: notebooks/transition_model_2D_airfoil
      :link-type: doc
      :class-card: tag-steady tag-rans tag-transition tag-aerospace tag-from_volume_mesh
      
      .. image:: thumbnails/NLF_airfoil.png
         :class: gallery-img

   .. grid-item-card:: UDD Alpha Controller (Constant CL)
      :link: notebooks/udd_alpha_controller
      :link-type: doc
      :class-card: tag-steady tag-rans tag-udd tag-aerospace tag-from_volume_mesh

      .. image:: thumbnails/om6_wing_vm.png
         :class: gallery-img

   .. grid-item-card:: Unsteady 2D Cylinder
      :link: notebooks/unsteady_2D_cylinder
      :link-type: doc
      :class-card: tag-unsteady tag-laminar tag-from_volume_mesh
      
      .. image:: thumbnails/unsteady_2D_cylinder.png
         :class: gallery-img

   .. grid-item-card:: Unsteady DDES HLPW5
      :link: notebooks/DDES_HLPW5
      :link-type: doc
      :class-card: tag-unsteady tag-ddes tag-aerospace tag-from_volume_mesh

      .. image:: thumbnails/HLPW5_geo.png
         :class: gallery-img

   .. grid-item-card:: Wall Roughness
      :link: notebooks/wall_roughness
      :link-type: doc
      :class-card: tag-steady tag-rans tag-aerospace tag-roughness tag-report tag-from_volume_mesh

      .. image:: thumbnails/wall_roughness.png
         :class: gallery-img

   .. grid-item-card:: Warm-Started Alpha Sweep
      :link: notebooks/warm_started_sweep
      :link-type: doc
      :class-card: tag-steady tag-rans tag-sweep tag-aerospace tag-from_geometry tag-new tag-stopping-criteria

      .. image:: thumbnails/warm_start_sweep.png
         :class: gallery-img

   .. grid-item-card:: Windsor Body
      :link: notebooks/windsor_body
      :link-type: doc
      :class-card: tag-steady tag-rans tag-automotive tag-from_geometry tag-geometry_ai
      
      .. image:: thumbnails/windsor_flow.png
         :class: gallery-img

.. toctree::
   :hidden:
   
   notebooks/2D_CRM_airfoil
   notebooks/aeroacoustics
   notebooks/alpha_sweep
   notebooks/BET_eVTOL
   notebooks/conjugate_heat_transfer
   notebooks/cube_snappy
   notebooks/DARPA_SUBOFF_AD
   notebooks/DDES_HLPW5
   notebooks/dynamic_derivatives
   notebooks/F1_car_demo
   notebooks/flat_plate_spring
   notebooks/hinge_torques
   notebooks/isolated_wing_BET
   notebooks/MRF_rotor
   notebooks/periodic_bc
   notebooks/time_accurate_wind_turbine
   notebooks/time_accurate_XV_15
   notebooks/transition_model_2D_airfoil
   notebooks/udd_alpha_controller
   notebooks/unsteady_2D_cylinder
   notebooks/wall_roughness
   notebooks/warm_started_sweep
   notebooks/windsor_body
