# Introduction

*An introduction to the basic features of the Flow360 WebUI including navigation and key concepts.*

## Contents

| *Section* | *Contents* |
|-------------|--------------|
|[Dashboard](./01.dashboard.md) | Navigation manual through the Flow360 Dashboard |
|[Starting a project](./02.starting-project.md) | Guidelines on how to start a project |
|[Workbench layout](./03.workbench-layout/README.md) | The layout of the project workbench |
|[General workflow](./04.general-workflow.md)| The example of a general workflow in the application |
|[Project tree](./05.project-tree.md)| Description of the project tree and the concept of a project |
|[Keyboard shortcuts](./06.shortcuts.md)| Available keyboard shortcuts in the Flow360 WebUI |
|[Project settings](./07.project-settings.md)| Configuration and metadata for projects and simulation runs |

---
## Detailed Descriptions

### Dashboard

*The Dashboard serves as the central hub for managing and monitoring your CFD projects in Flow360. It provides an intuitive overview of your simulations, resources, and project status.*

- **Recent Projects:** Grid view of recent CFD projects with status indicators, project details, and quick actions.
- **FlexCredits:** Displays your available computation credits.
- **Pending Resources:** Overview of computational resources in use, including running, queued, diverged, or failed jobs.
- **Start with the Example:** Quick-start option for new users with pre-prepared simulation scenarios.

---

### Starting a Project

*This section guides you through creating a new project in Flow360, whether starting from geometry, surface mesh, or volume mesh.*

- **New Project Form:** Entry point for creating CFD simulations, allowing selection of workflow type, solver version, units, and description.
- **Creation Methods:**
  - From geometry: Upload CAD files for automated meshing.
  - From surface mesh: Use a pre-generated surface mesh.
  - From volume mesh: Use a pre-generated volume mesh.
- **Supported Formats:** ESP, EGADS, STEP, IGES, ACIS, STL, CGNS, and more.
- **Key Fields:** Project name, solver version, unit, description, and creation method.

---

### Workbench Layout

*The Flow360 workbench provides an interface for CFD simulations, with defined regions for setup, visualization, and analysis.*

- **Viewer Region:** Central 3D workspace for geometry, mesh, and results visualization.
- **Simulation Setup/Analysis Panel:** Control panel for all simulation parameters and settings.
- **Entities Browser:** Tools for visual representation and inspection of model and results.
- **Coordinate System:** Persistent indicator of axes orientation and view direction.
- **Top Navigation Bar:** Main navigation and tool selection area.
- **Bottom Status Bar:** Displays operation status, progress, and inspector tools.
- **Viewer Bar:** Switches between geometry, mesh, and visualization modes, also contains entity selection tools.

---

### General Workflow

*This section outlines the systematic approach to setting up and running CFD simulations using the Flow360 GUI.*

- **Setup:** Configure simulation parameters in a top-to-bottom sequence.
- **Verify:** Use the inspector tool to check setup completeness and resolve errors.
- **Run:** Initiate the simulation, including automated meshing if starting from geometry.
- **Monitor:** Track simulation progress and analyze results in real time.
- **Visualize:** Inspect mesh and results using the interactive 3D viewer.

---

### Project Tree

*The project tree is a hierarchical visualization of your CFD workflow, from geometry to results. It helps track progress and navigate between stages.*

- **Icons:** Represent geometry, surface mesh, volume mesh, case, fork, and draft states.
- **Structure:** Flexible hierarchy supporting geometry-first, surface-mesh-first, and volume-mesh-first workflows.
- **Branching:** Supports multiple branches at each level for mesh sensitivity, parameter studies, and continuation runs (forks).
- **Component Actions:** Open in workbench, create new run, interpolate, rename, view details, expand/collapse, delete.

---

### Project Settings

*The Project Settings panel provides access to project-level configuration, run settings, and project metadata. This panel allows you to manage project properties, configure units, and view project statistics.*

- **Run Settings:** Configuration options for the current simulation run, including units system, mesher options, solver version, and run statistics.
- **Project Details:** Project-level metadata including project name, ID, folder location, tags, description, and project cost.
- **Units Management:** Change all units in the setting system at once through the units system configuration.
- **Mesher Options:** Toggle between standard and AI-powered surface meshing, and enable beta mesher features.
- **Statistics:** View detailed timing information, FlexCredit costs, and status for runs and projects.

```{toctree}
:hidden:
:maxdepth: 3
./01.dashboard.md
./02.starting-project.md
./03.workbench-layout/README.md
./04.general-workflow.md
./05.project-tree.md
./06.shortcuts.md
./07.project-settings.md
```