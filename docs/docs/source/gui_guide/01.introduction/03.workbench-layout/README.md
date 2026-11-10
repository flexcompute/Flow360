# Workbench Layout

*The Flow360 workbench provides an intuitive interface for computational fluid dynamics (CFD) simulations. This document describes the key areas of the workbench layout and their functions.*

![Workbench Layout Overview](../Figures/workbench-diagram-25.8.png)

---

## Main Interface Sections

| *Section* | *Description* |
|-------------|-----------------|
| 1 | [Viewer region](./05.viewer-region.md) |
| 2 | [Simulation setup](./../../02.simulation-setup/README.md) / [Analysis](./../../03.analysis/README.md) |
| 3 | [Entities browser](../../04.entities-browser/README.md) |
| 4 | Coordinate system |
| 5 | [Top navigation bar](./04.top-bar.md) |
| 6 | [Bottom status bar](./07.status-bar.md) |
| 7 | [Viewer bar](./05.viewer-region.md) | 

```{seealso}
For the workflow and interface overview, see the {doc}`User Guide: Workflows & Interfaces </user_guide/WorkflowsInterfaces/WorkflowsInterfaces>`.
```

---

## Detailed Descriptions

### **Viewer region**

*The central workspace where the geometry, mesh, and simulation results are displayed.* 

**This interactive 3D viewport allows you to:**
- Rotate (hold left mouse button), pan (hold right mouse button), and zoom the model (use the scroll)
- Select and inspect geometric features (left click)
- Preview created entities such as volumes or slices
- View mesh details
- Visualize simulation results and flow fields

```{tip}
Click right-mouse button while hovering in the model view area to see additional settings.
```

#### **Simulation setup / Analysis**

*A comprehensive control panel containing all simulation parameters and settings.*

**In this panel you will:**

- Type in your meshing parameters
- Set up the simulation's physics
- Change the solver numerics
- Define anticipated outputs
- Analyze and monitor the solution
- Visualise the results

#### **Entities browser**

*Dedicated controls for visual representation of the model and results.*
- Geometry display options
- Mesh visualization settings
- Results visualization tools
- Display modes and rendering options

#### **Coordinate system**

*Persistent coordinate system.*

- X, Y, and Z axes orientation
- Current view direction

#### **Top navigation bar**

*Primary navigation and tool selection area.*

**Actions available through the top bar:**
- Return to Flow360 dashboard
- Project tree
- Information about the current selected asset
- More button
- View only information (optional)
- Assets
- Help
- Project settings
- Fork / run case

#### **Bottom status bar**

*Information and status display*

- Current operation status
- Progress indicators
- Inspector tools
- Run status information

#### **Viewer bar**

*Allows you to switch between different modes as well as choosing selection options.*

The possible view modes depend on the [project creation method](../02.starting-project.md) and are:
- Geometry: presents the geometry of the simulated object.
- Mesh: shows the surface mesh generated on the surfaces
- Visualisation: visualises the flow field according to the chosen criteria (eg. Cp contour on surface)

The selection methods can be checked and unchecked and allow for the selection of following entities:
- Points
- Edges 
- Faces 
- Bodies

```{toctree}
:hidden:
:maxdepth: 3
./04.top-bar.md
./05.viewer-region.md
./07.status-bar.md
```