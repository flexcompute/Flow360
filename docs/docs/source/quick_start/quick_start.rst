.. _quickstart:

Quick Start
===========

This Quickstart guide walks you through the essentials of using Flow360.  
Choose between the **WebUI** or **Python API** paths below depending on your preferred workflow.

WebUI
-----

The WebUI Quickstart covers the end-to-end process of setting up, running, and analyzing a simulation directly from your browser.  
It’s designed for users who prefer an interactive, visual workflow - from uploading geometry and generating meshes to launching cases and viewing results.  
Each short tutorial focuses on a specific step in the process.

.. raw:: html

    <div class="cards-grid">

      <!-- Card Item 1 -->
      <div class="card-item">
        <a href="WebUI_Geometry/WebUI_Geometry.html" class="card-link">
          <div class="image-card">
            <img src="../_images/cad_cover.jpg"
                 alt="Geometry Upload Example"
                 class="card-img" />
          </div>
          <h3>1.1 WebUI Geometry Uploading</h3>
          <div class="card-meta">
            <span class="platform">WebUI</span>
            <span class="duration">8 minutes</span>
          </div>
        </a>
      </div>

      <!-- Card Item 2 -->
      <div class="card-item">
        <a href="WebUI_AutomatedMeshing/WebUI_AutomatedMeshing.html" class="card-link">
          <div class="image-card">
            <img src="../_images/mesh_cover.jpg"
                 alt="Automated Meshing Example"
                 class="card-img" />
          </div>
          <h3>1.2 WebUI Automated Meshing</h3>
          <div class="card-meta">
            <span class="platform">WebUI</span>
            <span class="duration">3 minutes</span>
          </div>
        </a>
      </div>

      <!-- Card Item 3 -->
      <div class="card-item">
        <a href="WebUI_CaseLaunching/WebUI_CaseLaunching.html" class="card-link">
          <div class="image-card">
            <img src="../_images/case_launching_cover.jpg"
                 alt="Case Launch Example"
                 class="card-img" />
          </div>
          <h3>1.3 WebUI Case Launching</h3>
          <div class="card-meta">
            <span class="platform">WebUI</span>
            <span class="duration">5 minutes</span>
          </div>
        </a>
      </div>

      <!-- Card Item 4 -->
      <div class="card-item">
        <a href="WebUI_CasePostprocessing/WebUI_CasePostprocessing.html" class="card-link">
          <div class="image-card">
            <img src="../_images/postprocess_cover.jpg"
                 alt="Postprocessing Example"
                 class="card-img" />
          </div>
          <h3>1.4 WebUI Case Postprocessing</h3>
          <div class="card-meta">
            <span class="platform">WebUI</span>
            <span class="duration">12 minutes</span>
          </div>
        </a>
      </div>

    </div>

Python API
----------

The Python API Quickstart introduces a fully scriptable way to run Flow360 simulations programmatically.  
It’s ideal for automation, parametric studies, or integrating Flow360 into larger Python workflows.  
You’ll learn how to create projects, launch cases, and extract results all from a few lines of code.

.. raw:: html

    <div class="cards-grid">

      <!-- Card Item 5 -->
      <div class="card-item">
        <a href="API_quickstart/notebooks/quickstart_API.html" class="card-link">
          <div class="image-card">
            <img src="../_images/api_cover.jpg"
                 alt="API Example"
                 class="card-img" />
          </div>
          <h3>1.5 Python API Quick Start</h3>
          <div class="card-meta">
            <span class="platform">Python API</span>
            <span class="duration">5 minutes</span>
          </div>
        </a>
      </div>

    </div>

.. figure:: ./WebUI_CasePostprocessing/Figures/postprocess_cover.jpg
   :width: 0%

.. figure:: ./WebUI_CaseLaunching/Figures/case_launching_cover.jpg
   :width: 0%

.. figure:: ./WebUI_AutomatedMeshing/Figures/mesh_cover.jpg
   :width: 0%

.. figure:: ./WebUI_Geometry/Figures/cad_cover.jpg
   :width: 0%

.. figure:: ./API_quickstart/Figures/api_cover.jpg
   :width: 0%


.. toctree::
    :maxdepth: 1
    :hidden:

    WebUI_Geometry/WebUI_Geometry
    WebUI_AutomatedMeshing/WebUI_AutomatedMeshing
    WebUI_CaseLaunching/WebUI_CaseLaunching
    WebUI_CasePostprocessing/WebUI_CasePostprocessing
    API_quickstart/notebooks/quickstart_API
