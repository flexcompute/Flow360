# Mesh 
A section for defining the meshing settings.

## Contents

| *Settings group* | *Description* |
|--------------------|-----------------|
|[**Farfield**](./01.farfield.md) | Outer boundaries and domain configuration |
|[**Mesh (meshing defaults)**](./02.mesh-parameters.md) | Fundamental control parameters for mesh generation | 
|[**Rotation zones**](./03.rotation-zones.md) | Specialized regions for rotating flow features |
|[**Custom zones**](./04.custom-zones.md) | User-defined volume zones |
|[**Volume mesh slices**](./05.volume-mesh-slices.md) | Slice outputs from the volume mesh |
|[**Refinements**](./06.refinements/README.md)| Local mesh control for critical regions |

---

## Detailed Descriptions

### [Farfield](./01.farfield.md)

*Defines the outer boundaries of your computational domain. Includes automated farfield, user-defined farfield, and wind tunnel configurations with various floor types.*

### [Mesh (meshing defaults)](./02.mesh-parameters.md)

*Fundamental control parameters for mesh generation affecting both surface and volume mesh characteristics. Provides global defaults for element size progression, boundary layer characteristics, surface mesh properties, and overall mesh quality control.*

### [Rotation zones](./03.rotation-zones.md)

*Specialized cylindrical volume zones designed for regions containing rotating components. Ensures proper mesh refinement and enables accurate simulation of rotating machinery with concentric mesh structure ideal for rotating components.*

### [Custom zones](./04.custom-zones.md)

*User-defined volume zones created from custom volumes or seedpoint volumes for specialized meshing regions.*

### [Volume mesh slices](./05.volume-mesh-slices.md)

*Configuration for extracting slice outputs from the generated volume mesh.*

### [Refinements](./06.refinements/README.md)

*Comprehensive local mesh control system for critical regions. Provides seven refinement types.*

- **[Surface Edge Refinement](./06.refinements/01.surface-edge-refinement.md)** - Controls mesh resolution near edges
- **[Surface Refinement](./06.refinements/02.surface-refinement.md)** - Controls surface mesh cell size
- **[Boundary Layer Refinement](./06.refinements/03.boundary-layer-refinement.md)** - Creates prismatic layers near walls
- **[Passive Spacing](./06.refinements/04.passive-spacing.md)** - Controls mesh behavior without direct refinement
- **[Uniform Refinement](./06.refinements/05.uniform-refinement.md)** - Creates uniform mesh spacing in a region
- **[Axisymmetric Refinement](./06.refinements/06.axisymmetric-refinement.md)** - Creates structured-like mesh with cylindrical bias
- **[Geometry Refinement](./06.refinements/07.geometry-refinement.md)** - Controls mesh resolution based on geometric features

```{toctree}
:hidden:
:maxdepth: 3
./01.farfield.md
./02.mesh-parameters.md
./03.rotation-zones.md
./04.custom-zones.md
./05.volume-mesh-slices.md
./06.refinements/README.md
```

