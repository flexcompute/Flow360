# Geometry

*Geometry is used to define the physical domain of the simulation in Flow360. Consequently, generation of surface and volume mesh will be based on the provided geometry.*


## Interacting with the geometry

In Flow360 you can do the following with geometry:

### Views
Flow360 provides three different ways to view and organize geometric entities using the view buttons at the top of the Geometry group:

#### Grouping View
In this view, entities are organized by their functional groups:

- **Edges grouping** - Controls how edges are grouped based on tags/names from the imported geometry.
  - Click the icon to the right of "Edges grouping" to change the grouping.
  - This affects which edge groups are available in the geometry, combining edges with common tags.

- **Faces grouping** - Controls how faces are grouped based on tags/names from the imported geometry.
  - Click the icon to the right of "Faces grouping" to change the grouping.
  - This affects which face groups are available in the geometry, combining faces with common tags.
  - **Important**: This setting determines the face names used for boundary conditions and surface output settings in your simulation.

- **Edges** - Lists all edges in the geometry, optionally grouped by the selected grouping method.

- **Faces** - Lists all faces in the geometry, optionally grouped by the selected grouping method.
  - You can rename faces using the menu next to each face.
  - You can change the color of faces or revert to the default color using the same menu.

- **Body Groups** - Shows the grouping of bodies according to their association with the imported geometry file (CAD or mesh).
  - **Display and highlighting** - Hover the mouse over any body group to highlight all faces in that group.
  - **Actions menu** - Click on an actions menu next to body group to perform actions on the body group.
    - **Mirror** - Create a mirrored copy of the body group about a symmetry plane:
      - **Group Name** - Name for the mirrored body group (defaults to `<mirror>` prefix)
      - **Normal** - The normal direction (X, Y, Z) of the mirror plane
      - **Origin** - The origin point (X, Y, Z) of the mirror plane
  - **Properties menu** - Click on a body within a group to view and edit its properties:
    - **Name** - The name of the body (derived from the source file by default).
    - **Tag** - Assign an entity tag to the body for organization.
    - **Mesh exterior** - When checked, the mesh will be generated on the exterior of this body. Useful for controlling which side of the geometry is meshed.
    - **Style** - Choose between **Color** or **Material** rendering:
      - **Color** - Set a solid color using the color picker or hex code.
      - **Material** - Apply a material appearance for realistic visualization.

#### Tree View
In this view, entities are organized in a hierarchical structure:

- **Bodies** - Shows all bodies available in the geometry.
  - **Edges** - Shows the edges associated with each body.
  - **Faces** - Shows the faces belonging to each body.



### Selection
*Allows for interactive selection of geometric entities.*

There are 4 entity selection modes that define whether a given entity can be selected or not:
- Select point
- Select edge
- Select face
- Select volume

### Grouping
*Grouping is based on geometric attributes and allows for organizing multiple elements into a single group for easier handling and assignment.*

As described in the Views section, you can control how edges and faces are grouped using the "Edges grouping" and "Faces grouping" options in the Grouping view. Proper grouping of geometric entities facilitates:

- Efficient assignment of boundary conditions
- Organized visualization of complex geometries
- Simplified selection of related entities

### Visualization
*Geometric entities are visualized in real-time. You can toggle on/off the visibility of grouped edges and faces.*

### Statistics
To view detailed statistics about your geometry, click the **Geometry Stats** button located to the right of the "Geometry" group heading. This opens a dialog displaying comprehensive statistics for the geometry in your project, including:
- Number of faces and edges
- Surface area calculations
- Other relevant geometric metrics


## Geometric Entities

Flow360 recognizes two primary types of geometric entities:

### [Edges](./01.edges.md)
Edges represent geometric elements that form the boundaries of faces. They are crucial for:
- Defining mesh refinement regions
- Controlling anisotropic layer growth
- Marking critical geometric features

### [Faces](./02.faces.md)
Faces are surfaces that constitute the boundaries of the computational domain. They serve multiple purposes:
- Specifying boundary conditions
- Defining regions for mesh refinement
- Organizing geometric features into logical groups


## Geometric Considerations

When preparing a simulation, keep in mind:

- **Mesh Generation**: The shape of geometry as well as its level of detail influences the quality and efficiency of the resulting mesh
- **Boundary Conditions and refinements**: Proper grouping of faces and edges makes assigning boundary conditions and local mesh refinements a much easier task.

```{toctree}
:hidden:
:maxdepth: 3
./01.edges.md
./02.faces.md
```