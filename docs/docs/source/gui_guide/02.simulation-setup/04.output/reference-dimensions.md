# Reference Dimensions

*The reference geometry parameters define essential dimensional characteristics used for computing aerodynamic coefficients and non-dimensional quantities in Flow360 simulations.*

---

## Available Parameters

| *Parameter* | *Description* |
|-------------|---------------|
| **Moment reference center** | Point coordinates defining the reference location for moment calculations |
| **Moment length** | Reference dimensions used for non-dimensionalization of moment coefficients |
| **Area** | Reference area used for non-dimensionalization of force and moment coefficients |

```{seealso}
For how reference dimensions are used to non-dimensionalize forces and moments, see the {doc}`/user_guide/NonDimensionalization/NonDimensionalization` user guide.
```

---

## Detailed Descriptions

### Moment reference center

*The moment reference center specifies the point about which aerodynamic moments are computed. This point is typically located at a significant position on the aircraft, such as the center of gravity or a specific percentage of the mean aerodynamic chord.*

- **Default:** `(0, 0, 0) m`
- **Example:** `(0.25, 0.0, 0.0) m`
>**Note:** The coordinate system follows the right-hand rule convention.

### Moment length

*Reference lengths used in the calculation of non-dimensional moment coefficients. These values are typically related to the characteristic dimensions of the geometry.*

- **Default:** `(1, 1, 1) m`
- **Example:** `(1.0, 5.0, 1.0) m`
>**Note:** Each component corresponds to the respective axis of rotation.

### Area

*The reference area is used to non-dimensionalize force and moment coefficients. For aircraft applications, this is typically the wing planform area, whereas for automotive cases it would be the frontal area of the car.*

- **Default:** `1 m²`
- **Example:** `10.0 m²`
>**Note:** Must be a positive non-zero value.

#### Automatic projected area

Python and the Web user interface can store an automatic projected-area recipe
in `ReferenceGeometry.area`. The recipe selects Geometry surfaces, a global
projection direction (`X`, `Y`, or `Z`), and raster quality. It is recomputed
before submission and retains both the recipe and latest computed value in the
uploaded simulation JSON.

Calling `fl.measure.projected_area(...)` is different: it computes and returns a
concrete preview immediately without changing simulation parameters. Both paths
apply active coordinate-system transformations and calculate rasterized
silhouette coverage, not bounding-box area or the sum of overlapping triangle
areas.

```{seealso}
See {doc}`/python_api/grab_and_go_snippets/compute_projected_area` for a Python
example and quality settings.
```

---

<details>
<summary><h3 style="display:inline-block"> 💡 Tips</h3></summary>

- The moment reference center location significantly impacts moment coefficient calculations.
- For aircraft analysis:
  - Reference area is typically the wing planform area
  - Moment reference center is often at 25% of the mean aerodynamic chord
  - Longitudinal (roll) moment length is usually the wingspan
  - Lateral (pitch) moment length is typically the mean aerodynamic chord
  - Vertical (yaw) moment length is often the mean aerodynamic chord
- For automotive analysis:
  - Reference area is typically the frontal area of the vehicle
  - Moment reference center is often at the center of the wheelbase at ground level
  - Longitudinal moment length is usually the wheelbase
  - Lateral moment length is typically the track width
  - Vertical moment length is commonly the vehicle height

</details>

---

<details>
<summary><h3 style="display:inline-block"> ❓ Frequently Asked Questions</h3></summary>

- **How do reference geometry parameters affect coefficient calculations?**  
  > These parameters are used to non-dimensionalize forces and moments into coefficient form. The reference area is used for all coefficient calculations, while moment lengths are used specifically for moment coefficients.

- **What happens if I use incorrect reference values?**  
  > Using incorrect reference values will result in incorrectly scaled coefficients, making it difficult to compare results with other simulations or experimental data.

- **Can I change reference values between simulations?**  
  > Yes, but ensure you account for the changes when comparing results between different simulations.

</details>

---

<details>
<summary><h3 style="display:inline-block"> 🐍 Python Example Usage</h3></summary>

```{seealso}
Python API reference: {py:class}`~flow360.ReferenceGeometry`.
```

```python
import flow360 as fl

# Define reference geometry parameters
with fl.SI_unit_system:
    params = fl.SimulationParams(
        reference_geometry=fl.ReferenceGeometry(
            moment_center=(0.25, 0.0, 0.0),
            moment_length=(1.0, 5.0, 1.0),
            area=10.0
        )
    )
```

</details> 
