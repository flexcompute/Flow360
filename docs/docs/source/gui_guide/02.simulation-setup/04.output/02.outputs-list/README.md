# Outputs List

*Control what data is generated during and after simulation runs, including which flow variables are saved, where they are saved, and how frequently they are written to files.*

---

## Available Output Types

| *Output Type* | *Description* | *Use Case* |
|---------------|---------------|------------|
| **[Volume Output](./01.volume-output.md)** | Flow field data throughout the computational volume | Visualizing 3D flow structures, vortex development |
| **[Time-averaging Volume Output](./02.time-averaging-volume-output.md)** | Time-averaged flow field data throughout the volume | Statistical analysis of unsteady flows |
| **[Surface Output](./03.surface-output.md)** | Flow field data on geometry or volume mesh boundaries | Analyzing surface forces, pressure distributions |
| **[Time-averaging Surface Output](./04.time-averaging-surface-output.md)** | Time-averaged flow field data on surfaces | Statistical analysis of surface phenomena |
| **[Slice Output](./05.slice-output.md)** | Flow field data on user-defined slice planes | Examining flow cross-sections |
| **[Time-averaging Slice Output](./06.time-averaging-slice-output.md)** | Time-averaged flow field data on slice planes | Statistical analysis on planes |
| **[Probe Outputs](./07.probe-outputs.md)** | Flow field data monitoring during simulation | Tracking convergence and flow properties |
| **[Time-averaging Probe Outputs](./08.time-averaging-probe-outputs.md)** | Time-averaged monitoring data | Statistical monitoring data |
| **[Surface Probe Output](./09.surface-probe-outputs.md)** | Flow field data at specific points projected onto surfaces | Point-specific surface data |
| **[Surface Integral Output](./20.surface-integral-output.md)** | Surface integral of custom user variables on selected surfaces | Integrated forces, torques, and custom derived quantities |
| **[Isosurface Output](./11.isosurface-output.md)** | Flow field data on surfaces of constant variable value | Visualization of vortices, shock waves |
| **[Time-averaging Isosurface Output](./12.time-averaging-isosurface-output.md)** | Time-averaged flow field data on surfaces of constant variable value | Statistical analysis of vortices, shock waves |
| **[Aeroacoustic Output](./13.aeroacoustic-output.md)** | Data for aeroacoustic analysis at observer positions | Noise prediction and analysis |
| **[Streamline Output](./14.streamline-output.md)** | Streamline visualization data | Visualization of 3D flow structures |
| **[Time-averaging Streamline Output](./15.time-averaging-streamline-output.md)** | Time-averaged streamline visualization data | Statistical analysis of flow structures |
| **[Render Output](./21.render-output.md)** | Rendered images of the solution produced by the solver | Presentation images without external post-processing |
| **[Force Output](./16.force-output.md)** | Force and moment coefficient outputs with optional statistics | Tracking aerodynamic performance, monitoring convergence |
| **[Force Distribution Output](./17.force-distribution-output.md)** | Custom force and moment distribution along a specified direction | Spanwise/chordwise loading analysis, directional force analysis |
| **[Time-averaging Force Distribution Output](./18.time-averaging-force-distribution-output.md)** | Time-averaged custom force and moment distribution | Statistical analysis of unsteady force distributions |

---

## Additional Outputs Available Through Python API

| *Output Type* | *Description* | *Use Case* |
|---------------|---------------|------------|
| **[Surface Slice Output](./10.surface-slice-output.md)** | Flow field data on slices of surfaces | Cross-sectional analysis of surface phenomena |

---

## Reference Documentation

- **[Available Output Fields](../output-fields.md)** - Complete list of flow variables by output type
- **[Scaling Values and Nondimensionalization](../scaling-values.md)** - Reference values for converting to physical units
- **[Output Formats](19.output-formats.md)** - Supported file formats (ParaView, Tecplot, CSV)

For configuration details specific to each output type, see the individual output pages linked in the table above.

---

<details>
<summary><h3 style="display:inline-block"> ❓ Frequently Asked Questions</h3></summary>

- **How can I visualize vortices in my flow?**  
  > Use Volume Output with Q-criterion field or create dedicated Isosurface Output of Q-criterion.

- **What's the difference between Surface Output and Surface Probe Output?**  
  > Surface Output captures data across entire surfaces, while Surface Probe Output captures data at specific points on surfaces.

- **Can I save outputs in different formats?**  
  > Yes. The **Output format** control is multi-select: choose any combination of Paraview (`.vtu`, `.vtp`), Tecplot (`.szplt`), VTK-HDF (`.vtkhdf`), and EnSight (`.case`, `.geo`). See [Output Formats](19.output-formats.md) for details.

</details>

---

```{toctree}
:hidden:
:maxdepth: 3
./01.volume-output.md
./02.time-averaging-volume-output.md
./03.surface-output.md
./04.time-averaging-surface-output.md
./05.slice-output.md
./06.time-averaging-slice-output.md
./07.probe-outputs.md
./08.time-averaging-probe-outputs.md
./09.surface-probe-outputs.md
./10.surface-slice-output.md
./20.surface-integral-output.md
./11.isosurface-output.md
./12.time-averaging-isosurface-output.md
./13.aeroacoustic-output.md
./14.streamline-output.md
./15.time-averaging-streamline-output.md
./21.render-output.md
./16.force-output.md
./17.force-distribution-output.md
./18.time-averaging-force-distribution-output.md
./19.output-formats.md
```