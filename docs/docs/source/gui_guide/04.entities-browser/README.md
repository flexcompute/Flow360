# Entities browser

*Entities are the fundamental components used for simulation setup in Flow360. They provide a hierarchical structure for organizing and managing CFD simulations, from geometry definition to results analysis.*

**Showing/hiding the entities browser**: The Entities browser is shown by clicking the button in the top-right corner of the viewer area.  

**Entities visibility, highlighting (hover)**: As you hover the mouse over items in the Entities browser, they are highlighted in the viewer.  

**Entities browser modes**: The Entities browser shows the Geometry, Mesh or Visualization items currently available in the viewer, and so it changes when the View mode is selected.

**"View only" mode**: The Entities browser has limited functionality when the Workbench is in "View only" mode. It can be used to show/hide elements and to browse the available entities and highlight them in the viewer. Creating and updating viewpoints is also supported at any time.

> **Draft Configuration**: Modifications made in the Entities browser are saved as a **Draft** configuration. This draft can be retrieved and continued from the Project Tree view, allowing you to resume your work at any time.


## Primary Assets

The primary asset defines the starting point of your simulation workflow. Only one primary asset type is available per project, depending on how the project was created.

### ![](./Figures/geometry.svg) [**Geometry**](./01.geometry/README.md)
The root entity for CAD-based projects. Contains all geometric elements including edges, faces, and body groups. Geometry projects progress through surface meshing and volume meshing before simulation.

### ![](./Figures/surface-mesh.svg) [**Surface Mesh**](./02.surface-mesh.md)
The root entity for surface-mesh-based projects. Contains boundary definitions for specifying boundary conditions. Surface mesh projects require volume meshing before simulation.

### ![](./Figures/volume-mesh.svg) [**Volume Mesh**](./03.volume-mesh/README.md)
The root entity for volume-mesh-based projects. Contains zones and boundaries ready for simulation setup. Volume mesh projects can proceed directly to solver configuration.


## Geometric Entities

These entities define spatial regions and monitoring locations within your simulation domain. Their availability depends on the primary asset type.

| Entity | Description | Geometry | Surface Mesh | Volume Mesh |
|--------|-------------|:--------:|:------------:|:-----------:|
| [**Body group**](./01.geometry/README.md) | Grouping of bodies from imported geometry files | ✓ | — | — |
| [**Volume mesh slices**](./03.volume-mesh/README.md) | Cross-sections of the volume mesh for inspection | — | — | ✓ |
| [**Volumes**](./04.volumes/README.md) | 3D regions for mesh refinement and flow analysis | ✓ | ✓ | ✓ |
| [**Points**](./05.points.md) | Monitoring locations for flow properties | ✓ | ✓ | ✓ |
| [**Slices**](./06.slices.md) | Planar cross-sections for flow visualization | ✓ | ✓ | ✓ |
| [**Sample Surfaces**](./10.sample-surfaces.md) | Imported surfaces for flow data extraction | ✓ | ✓ | ✓ |
| [**Derived**](./07.derived.md) | Auto-generated entities (farfield, symmetry planes) | ✓ | ✓ | — |


## Configuration Tools

These entities are available in all project types and provide tools for visualization, organization, and scene management.

### [**Viewpoints**](./08.viewpoints.md)
Saved camera positions and orientations for consistent visualization across sessions. Viewpoints can be created and modified even in "View only" mode.

### [**Coordinate Systems**](./11.coordinate-systems.md)
Local reference frames for geometric transformations. Define custom coordinate systems with translation, rotation, and scaling relative to the global frame or other coordinate systems.

### [**Environment**](./09.environment.md)
Scene and background customization settings for the viewer, including background colors, gradients, and image uploads.

### [**Entity Tags**](./12.entity-tags.md)
Custom labels for organizing and grouping entities. Tags support expression-based filtering for automatic entity selection.


```{seealso}
- [Geometry](./01.geometry/README.md)
- [Surface Mesh](./02.surface-mesh.md)
- [Volume Mesh](./03.volume-mesh/README.md)
- [Volumes](./04.volumes/README.md)
- [Points](./05.points.md)
- [Slices](./06.slices.md)
- [Sample Surfaces](./10.sample-surfaces.md)
- [Derived](./07.derived.md)
- [Viewpoints](./08.viewpoints.md)
- [Coordinate Systems](./11.coordinate-systems.md)
- [Environment](./09.environment.md)
- [Entity Tags](./12.entity-tags.md)
```

```{toctree}
:hidden:
:maxdepth: 3
./01.geometry/README.md
./02.surface-mesh.md
./03.volume-mesh/README.md
./04.volumes/README.md
./05.points.md
./06.slices.md
./10.sample-surfaces.md
./07.derived.md
./08.viewpoints.md
./11.coordinate-systems.md
./09.environment.md
./12.entity-tags.md
```
