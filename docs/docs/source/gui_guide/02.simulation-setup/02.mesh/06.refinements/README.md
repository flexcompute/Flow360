# Refinements

*This document provides an overview of mesh refinement capabilities in Flow360. Refinements enable precise control over mesh resolution in specific regions of your geometry to better capture flow features or geometric details.*

## Available Refinement Types

| *Refinement type* | *Description* |
|--------------------|-----------------|
|[**Surface Edge Refinement**](./01.surface-edge-refinement.md) | Controls mesh resolution near edges |
|[**Surface Refinement**](./02.surface-refinement.md) | Controls surface mesh cell size |
|[**Boundary Layer Refinement**](./03.boundary-layer-refinement.md) | Creates prismatic layers near walls |
|[**Passive Spacing**](./04.passive-spacing.md) | Controls mesh behavior without direct refinement |
|[**Uniform Refinement**](./05.uniform-refinement.md) | Creates uniform mesh spacing in a region |
|[**Axisymmetric Refinement**](./06.axisymmetric-refinement.md) | Creates structured-like mesh with cylindrical bias |
|[**Geometry Refinement**](./07.geometry-refinement.md) | Controls mesh resolution based on geometric features |

```{seealso}
For a deeper discussion of the meshing workflow and parameters, see {doc}`/user_guide/Meshing/Meshing`.
```

---

<details>
<summary><h3 style="display:inline-block"> ❓ Frequently Asked Questions</h3></summary>

- **What happens if refinements overlap?**  
  > The finest (smallest) spacing will be used in overlapping regions.

</details>

---

<details>
<summary><h3 style="display:inline-block"> 🐍 Python Example Usage</h3></summary>

```{seealso}
Python API:
- {py:class}`~flow360.SurfaceEdgeRefinement`
- {py:class}`~flow360.SurfaceRefinement`
- {py:class}`~flow360.BoundaryLayer`
- {py:class}`~flow360.PassiveSpacing`
- {py:class}`~flow360.UniformRefinement`
- {py:class}`~flow360.AxisymmetricRefinement`
- {py:class}`~flow360.GeometryRefinement`
```

```python
import flow360 as fl

# Example of combining multiple refinements
meshing_params = fl.MeshingParams(
    refinements=[
        # Surface edge refinement for leading edge
        fl.SurfaceEdgeRefinement(
            name="leading_edge",
            edges=[leading_edge],
            method=fl.HeightBasedRefinement(value=0.001 * fl.u.m)
        ),
        # Surface refinement for general resolution
        fl.SurfaceRefinement(
            name="wing_surface",
            faces=[wing_surface],
            max_edge_length=0.05 * fl.u.m
        ),
        # Boundary layer refinement for wall regions
        fl.BoundaryLayer(
            name="wing_bl",
            faces=[wing_surface],
            first_layer_thickness=1e-5 * fl.u.m,
            growth_rate=1.2
        ),
        # Passive spacing refinement for interface region
        fl.PassiveSpacing(
            name="interface_region",
            type="projected",
            faces=[interface_surface]
        ),
        # Wake region refinement
        fl.UniformRefinement(
            name="wake_region",
            entities=[wake_box],
            spacing=0.1 * fl.u.m
        ),
        # Axisymmetric refinement for propeller region
        fl.AxisymmetricRefinement(
            name="propeller_region",
            entities=[prop_cylinder],
            spacing_axial=0.02 * fl.u.m,
            spacing_radial=0.01 * fl.u.m,
            spacing_circumferential=0.015 * fl.u.m
        ),
        # Geometry refinement for fine features
        fl.GeometryRefinement(
            name="fine_features_refinement",
            faces=[wing_surface, fuselage_surface],
            geometry_accuracy=0.001 * fl.u.m,
            preserve_thin_geometry=True
        )
    ]
)
```
</details> 


```{toctree}
:hidden:
:maxdepth: 3
./01.surface-edge-refinement.md
./02.surface-refinement.md
./03.boundary-layer-refinement.md
./04.passive-spacing.md
./05.uniform-refinement.md
./06.axisymmetric-refinement.md
./07.geometry-refinement.md
```