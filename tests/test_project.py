import pytest

import flow360 as fl
from flow360 import log
from flow360.exceptions import (
    Flow360DuplicateAssetError,
    Flow360FileError,
    Flow360ValueError,
    Flow360WebError,
)

log.set_logging_level("DEBUG")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_from_cloud(mock_id, mock_response):
    project = fl.Project.from_cloud(project_id="prj-41d2333b-85fd-4bed-ae13-15dcb6da519e")
    project.print_project_tree(is_horizontal=True, str_length=15)

    assert isinstance(project._root_asset, fl.Geometry)
    assert len(project.get_cached_surface_meshe_ids()) == 1
    assert len(project.get_cached_volume_meshe_ids()) == 1
    assert len(project.get_cached_case_ids()) == 2

    current_geometry_id = "geo-2877e124-96ff-473d-864b-11eec8648d42"
    current_surface_mesh_id = "sm-1f1f2753-fe31-47ea-b3ab-efb2313ab65a"
    current_volume_mesh_id = "vm-7c3681cd-8c6c-4db7-a62c-1742d825e9d3"
    current_case_id = "case-84d4604e-f3cd-4c6b-8517-92a80a3346d3"

    assert project.geometry.id == current_geometry_id
    assert project.surface_mesh.id == current_surface_mesh_id
    assert project.volume_mesh.id == current_volume_mesh_id
    assert project.case.id == current_case_id

    project._case_cache.asset_cache = {}
    error_msg = "Cache is empty, no assets are available"
    with pytest.raises(Flow360ValueError, match=error_msg):
        project.get_case(asset_id=current_case_id)


def test_get_asset_with_id(mock_id, mock_response):
    project = fl.Project.from_cloud(project_id="prj-41d2333b-85fd-4bed-ae13-15dcb6da519e")
    print(f"The correct asset_id: {project.case.id}")

    query_id = "case-b0"
    error_msg = (
        r"The input asset ID \(" + query_id + r"\) is too short to retrive the correct asset."
    )
    with pytest.raises(Flow360ValueError, match=error_msg):
        project.get_case(asset_id=query_id)

    query_id = "case-1234567-abcd"
    error_msg = (
        r"This asset does not exist in this project. Please check the input asset ID \("
        + query_id
        + r"\)"
    )
    with pytest.raises(Flow360ValueError, match=error_msg):
        project.get_case(asset_id=query_id)


# @pytest.mark.usefixtures("s3_download_override")
# def test_get_params(mock_response):
#     project = fl.Project.from_cloud(project_id="prj-41d2333b-85fd-4bed-ae13-15dcb6da519e")
#     params_dict = project.case.params.dict()
#     print(params_dict)
#     with fl.SI_unit_system:
#         params = fl.SimulationParams(**params_dict)
#     print(params)

# volume_mesh = project.volume_mesh
# params_dict = fl.VolumeMesh._get_simulation_json(volume_mesh)
# params_dict["time_stepping"].pop("order_of_accuracy")
# with fl.SI_unit_system:
#     params = fl.SimulationParams(**params_dict)


def test_run(mock_response, capsys):
    project = fl.Project.from_cloud(project_id="prj-41d2333b-85fd-4bed-ae13-15dcb6da519e")
    geometry = project.geometry

    # show face groupings
    geometry.show_available_groupings(verbose_mode=True)
    geometry.group_faces_by_tag("faceName")
    geometry.group_edges_by_tag("edgeName")

    with fl.SI_unit_system:
        cylinder = fl.Cylinder(
            name="cylinder",
            axis=[0, 1, 0],
            center=[0, 0, 0],
            inner_radius=0,
            outer_radius=1.0,
            height=2.5,
        )
        sliding_interface = fl.RotationCylinder(
            spacing_axial=0.04,
            spacing_radial=0.04,
            spacing_circumferential=0.04,
            entities=cylinder,
            enclosed_entities=geometry["wing"],
        )
        farfield = fl.AutomatedFarfield(name="farfield")
        params = fl.SimulationParams(
            meshing=fl.MeshingParams(
                defaults=fl.MeshingDefaults(
                    surface_max_edge_length=0.03 * fl.u.m,
                    curvature_resolution_angle=8 * fl.u.deg,
                    surface_edge_growth_rate=1.15,
                    boundary_layer_first_layer_thickness=1e-6,
                    boundary_layer_growth_rate=1.15,
                ),
                refinement_factor=1.0,
                volume_zones=[sliding_interface, farfield],
                refinements=[
                    fl.SurfaceEdgeRefinement(
                        name="leadingEdge",
                        method=fl.AngleBasedRefinement(value=1 * fl.u.degree),
                        edges=geometry["leadingEdge"],
                    ),
                    fl.SurfaceEdgeRefinement(
                        name="trailingEdge",
                        method=fl.HeightBasedRefinement(value=0.001),
                        edges=geometry["trailingEdge"],
                    ),
                ],
            ),
            reference_geometry=fl.ReferenceGeometry(
                moment_center=[0, 0, 0],
                moment_length=[1, 1, 1],
                area=2,
            ),
            operating_condition=fl.AerospaceCondition(
                velocity_magnitude=50,
            ),
            time_stepping=fl.Steady(
                max_steps=10000, CFL=fl.RampCFL(initial=1, final=100, ramp_steps=1000)
            ),
            outputs=[
                fl.VolumeOutput(
                    name="VolumeOutput",
                    output_fields=[
                        "Mach",
                    ],
                ),
                fl.SurfaceOutput(
                    name="SurfaceOutput",
                    surfaces=geometry["*"],
                    output_fields=[
                        "Cp",
                        "CfVec",
                    ],
                ),
            ],
            models=[
                fl.Rotation(
                    volumes=cylinder,
                    spec=fl.AngularVelocity(0 * fl.u.rad / fl.u.s),
                ),
                fl.Freestream(surfaces=farfield.farfield, name="Freestream"),
                fl.Wall(surfaces=geometry["wing"], name="NoSlipWall"),
                fl.Fluid(
                    navier_stokes_solver=fl.NavierStokesSolver(
                        absolute_tolerance=1e-9,
                        linear_solver=fl.LinearSolver(max_iterations=35),
                    ),
                    turbulence_model_solver=fl.SpalartAllmaras(
                        absolute_tolerance=1e-8,
                        linear_solver=fl.LinearSolver(max_iterations=25),
                    ),
                ),
            ],
        )

    warning_msg = "We already submitted this Case in the project."
    project.run_case(params=params)
    captured_text = capsys.readouterr().out
    captured_text = " ".join(captured_text.split())
    assert warning_msg in captured_text


def test_generate_volume_mesh(mock_response, capsys):

    project = fl.Project.from_cloud(project_id="prj-41d2333b-85fd-4bed-ae13-15dcb6da519e")

    volume_mesh = project.volume_mesh
    params_dict = fl.VolumeMesh._get_simulation_json(volume_mesh)
    with fl.SI_unit_system:
        params = fl.SimulationParams(**params_dict)

    warning_msg = "We already generated this Volume Mesh in the project."
    project.generate_volume_mesh(params=params)
    captured_text = capsys.readouterr().out
    captured_text = " ".join(captured_text.split())
    assert warning_msg in captured_text
