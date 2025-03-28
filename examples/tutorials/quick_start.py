# Import necessary modules from the Flow360 library
import flow360 as fl
from flow360.examples import eVTOL
from flow360.log import log

# Create a project and upload the geometry
project = fl.Project.from_file(eVTOL.geometry, name="EVTOL_quickstart_from_API", length_unit="m")

log.info(f"The project id is {project.id}")

# Group faces and edges by their specific tags for easier assignment of simulation parameters.
# The tag names were assigned in the .csm script.
geometry = project.geometry
geometry.group_faces_by_tag("faceName")
geometry.group_edges_by_tag("edgeName")

# Create the farfield object.
farfield_zone = fl.AutomatedFarfield()

# Create a simulation params object using the SI unit system as default dimensions.
with fl.SI_unit_system:
    params = fl.SimulationParams(
        # Set the meshing parameters.
        meshing=fl.MeshingParams(
            defaults=fl.MeshingDefaults(
                boundary_layer_first_layer_thickness=1e-5, surface_max_edge_length=1
            ),
            volume_zones=[farfield_zone],
            # Set the refinement parameters for the leading edges.
            refinements=[
                fl.SurfaceEdgeRefinement(
                    name="leading_edges",
                    edges=[geometry["leadingEdge"]],
                    method=fl.AngleBasedRefinement(value=2 * fl.u.deg),
                )
            ],
        ),
        # Set the operating condition.
        operating_condition=fl.AerospaceCondition(
            velocity_magnitude=50 * fl.u.m / fl.u.s, alpha=0 * fl.u.deg
        ),
        # Set the time stepping parameters.
        time_stepping=fl.Steady(max_steps=3000),
        # Set the physics models.
        models=[
            # Assign each generated mesh patch as defined in the .csm file to their matching airplane subcomponents.
            fl.Wall(surfaces=geometry["fuselage"], name="fuselage"),
            # Notice below how we use the * operator to assign both left and right pylon to the pylons subcomponent
            fl.Wall(surfaces=geometry["*_pylon"], name="pylons"),
            fl.Wall(surfaces=geometry["*_wing"], name="wing"),
            fl.Wall(surfaces=geometry["*_tail"], name="tail"),
            fl.Freestream(surfaces=[farfield_zone.farfield], name="Freestream"),
        ],
        # Output format could be 'paraview' or 'tecplot' or 'both'.
        outputs=[
            fl.SurfaceOutput(
                # Select all surfaces for output
                surfaces=geometry["*"],
                # Output fields for post-processing
                output_fields=["Cp", "Cf", "yPlus", "CfVec"],
                output_format="both",
            )
        ],
    )
# Run the case.
project.run_case(params=params, name=f"EVTOL_quickstart_alpha{params.operating_condition.alpha}")
log.info(f"The case ID is: {project.case.id} with alpha = {params.operating_condition.alpha} ")
