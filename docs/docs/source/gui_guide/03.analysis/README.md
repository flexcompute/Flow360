# Analysis

*A comprehensive suite of tools for analyzing and visualizing simulation results in Flow360.*

---

## Available Options

| *Option* | *Description* | *Purpose* |
|------------|-----------------|-------------|
| [Dashboard](./01.dashboard.md) | Interactive overview of simulation progress and key metrics | Real-time monitoring and analysis |
| [Convergence](./02.convergence.md) | Analysis of solution convergence behavior | Solution quality assessment |
| [Monitor](./03.monitor.md) | Tracking of flow field variables at specific locations | Detailed flow analysis |
| [Visualization](./04.visualization.md) | Advanced visualization tools for flow field data | Flow pattern analysis |
| [Aeroacoustic](./05.aeroacoustic.md) | Acoustic analysis and noise prediction | Sound generation and propagation |

---

## Detailed Descriptions

### Dashboard

*The Dashboard provides an interactive overview of your simulation's progress and key performance metrics.*

**Features:**
- Real-time progress tracking
- Key performance indicators
- Simulation status overview
- Resource utilization metrics

### Convergence

*Convergence analysis tools help assess the quality and stability of your simulation results.*

**Features:**
- Nonlinear residual monitoring
- Linear residual tracking
- CFL number visualization
- State variable bounds
- Maximum residual location tracking

### Monitor

*Monitors enable detailed tracking of flow field variables at specific locations during simulation.*

**Available Types:**
- Total Forces
- Forces by Surface
- Heat Transfer by Surface
- BET Forces and Moments
- Force Distribution (X/Y)
- Actuator Disk metrics


### Visualization

*Advanced visualization tools for analyzing flow field data and patterns.*


**Features:**
- Surface visualization
- Volume visualization
- Slice analysis
- Isosurface generation
- Streamline visualization

### Aeroacoustic

*Specialized tools for acoustic analysis and noise prediction.*

**Features:**
- Acoustic pressure monitoring
- Frequency spectrum analysis
- Overall Sound Pressure Level (OASPL) calculation
- A-weighting support
- Multiple observer positions

---

```{toctree}
:hidden:
:maxdepth: 3
./01.dashboard.md
./02.convergence.md
./03.monitor.md
./04.visualization.md
./05.aeroacoustic.md
```