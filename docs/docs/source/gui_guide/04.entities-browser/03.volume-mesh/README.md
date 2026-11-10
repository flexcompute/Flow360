# Volume Mesh

*The volume mesh represents the discretized computational domain in Flow360, where fluid flow phenomena are numerically solved.*

A volume mesh in Flow360 consists of three-dimensional elements that discretize the computational domain. The mesh structure is defined by:

- [Zones](./01.zones.md): Distinct regions within the volume mesh, each potentially governed by different physical models
- [Boundaries](./02.boundaries.md): Surfaces that define the edges of computational domains and interfaces between zones

## Mesh Requirements

Flow360 supports unstructured volume meshes with the following element types:

- Tetrahedrons
- Hexahedrons
- Prisms
- Pyramids

```{toctree}
:hidden:
:maxdepth: 3
./01.zones.md
./02.boundaries.md
```