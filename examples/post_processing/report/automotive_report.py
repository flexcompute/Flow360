import flow360 as fl
from flow360 import u
from flow360.examples import DrivAer
from flow360.log import log
from flow360.plugins.report.report import ReportTemplate
from flow360.plugins.report.report_items import (
    BottomCamera,
    Chart2D,
    Chart3D,
    FrontCamera,
    FrontLeftBottomCamera,
    FrontLeftTopCamera,
    Inputs,
    LeftCamera,
    RearCamera,
    RearLeftTopCamera,
    RearRightBottomCamera,
    Settings,
    Summary,
    Table,
    TopCamera,
)
from flow360.plugins.report.utils import Average, DataItem, Delta, Expression, Variable
from flow360.version import __solver_version__

DrivAer.get_files()

project = fl.Project.from_volume_mesh(
    DrivAer.mesh_filename,
    name="Automotive DrivAer",
)

vm = project.volume_mesh

log.info("Volume mesh contains the following boundaries:")
for boundary in vm.boundary_names:
    log.info("Boundary: " + boundary)

freestream_surfaces = ["blk-1/WT_side1", "blk-1/WT_side2", "blk-1/WT_inlet", "blk-1/WT_outlet"]
slip_wall_surfaces = ["blk-1/WT_ceiling", "blk-1/WT_ground_front", "blk-1/WT_ground"]
wall_surfaces = list(set(vm.boundary_names) - set(freestream_surfaces) - set(slip_wall_surfaces))

cases = []

for beta in [0, 5, 10]:
    with fl.SI_unit_system:
        params = fl.SimulationParams(
            meshing=None,
            reference_geometry=fl.ReferenceGeometry(area=2.17, moment_length=2.7862),
            operating_condition=fl.AerospaceCondition(velocity_magnitude=40, beta=beta * u.deg),
            models=[
                fl.Wall(surfaces=[vm[i] for i in wall_surfaces], use_wall_function=True),
                fl.Freestream(
                    surfaces=[vm[i] for i in freestream_surfaces],
                ),
                fl.SlipWall(
                    surfaces=[vm[i] for i in slip_wall_surfaces],
                ),
            ],
            user_defined_fields=[
                fl.UserDefinedField(
                    name="Cpx",
                    expression="double prel = primitiveVars[4] - pressureFreestream;"
                    + "double PressureForce_X = prel * nodeNormals[0]; "
                    + "Cpx = PressureForce_X / (0.5 * MachRef * MachRef) / magnitude(nodeNormals);",
                ),
            ],
            outputs=[
                fl.SurfaceOutput(
                    surfaces=vm["*"],
                    output_fields=[
                        "Cp",
                        "Cf",
                        "yPlus",
                        "CfVec",
                        "primitiveVars",
                        "wall_shear_stress_magnitude",
                        "Cpx",
                    ],
                ),
                fl.SliceOutput(
                    entities=[
                        *[
                            fl.Slice(
                                name=f"slice_y_{name}",
                                normal=(0, 1, 0),
                                origin=(0, y, 0),
                            )
                            for name, y in zip(
                                ["0", "0_2", "0_4", "0_6", "0_8"], [0, 0.2, 0.4, 0.6, 0.8]
                            )
                        ],
                        *[
                            fl.Slice(
                                name=f"slice_z_{name}",
                                normal=(0, 0, 1),
                                origin=(0, 0, z),
                            )
                            for name, z in zip(
                                ["neg0_2", "0", "0_2", "0_4", "0_6", "0_8"],
                                [-0.2, 0, 0.2, 0.4, 0.6, 0.8],
                            )
                        ],
                    ],
                    output_fields=["velocity", "velocity_x", "velocity_y", "velocity_z"],
                ),
                fl.IsosurfaceOutput(
                    output_fields=["Cp", "Mach"],
                    isosurfaces=[
                        fl.Isosurface(
                            name="isosurface-cpt",
                            iso_value=-1,
                            field="Cpt",
                        ),
                    ],
                ),
                fl.ProbeOutput(
                    entities=[fl.Point(name="point1", location=(10, 0, 1))],
                    output_fields=["velocity"],
                ),
            ],
        )

    case_new = project.run_case(params=params, name=f"DrivAer 5.7M - beta={beta}")

    cases.append(case_new)

# wait until all cases finish running
for case in cases:
    case.wait()

exclude = ["blk-1/WT_ground_close", "blk-1/WT_ground_patch"]
size = "5.7M"

exclude += freestream_surfaces + slip_wall_surfaces

top_camera = TopCamera(pan_target=(1.5, 0, 0), dimension=5, dimension_dir="width")
top_camera_slice = TopCamera(pan_target=(2.5, 0, 0), dimension=8, dimension_dir="width")
side_camera = LeftCamera(pan_target=(1.5, 0, 0), dimension=5, dimension_dir="width")
side_camera_slice = LeftCamera(pan_target=(2.5, 0, 1.5), dimension=8, dimension_dir="width")
rear_camera = RearCamera(dimension=2.5, dimension_dir="width")
front_camera = FrontCamera(dimension=2.5, dimension_dir="width")
bottom_camera = BottomCamera(pan_target=(1.5, 0, 0), dimension=5, dimension_dir="width")
front_left_bottom_camera = FrontLeftBottomCamera(
    pan_target=(1.5, 0, 0), dimension=5, dimension_dir="width"
)
rear_right_bottom_camera = RearRightBottomCamera(
    pan_target=(1.5, 0, 0), dimension=6, dimension_dir="width"
)
front_left_top_camera = FrontLeftTopCamera(
    pan_target=(1.5, 0, 0), dimension=6, dimension_dir="width"
)
rear_left_top_camera = RearLeftTopCamera(pan_target=(1.5, 0, 0), dimension=6, dimension_dir="width")

cameras_geo = [
    top_camera,
    side_camera,
    rear_camera,
    bottom_camera,
    front_left_bottom_camera,
    rear_right_bottom_camera,
]

limits_cp = [(-1, 1), (-1, 1), (-1, 1), (-0.3, 0), (-0.3, 0), (-1, 1), (-1, 1), (-1, 1)]
cameras_cp = [
    front_camera,
    front_left_top_camera,
    side_camera,
    rear_left_top_camera,
    rear_camera,
    bottom_camera,
    front_left_bottom_camera,
    rear_right_bottom_camera,
]

avg = Average(fraction=0.1)
CD = DataItem(data="surface_forces/totalCD", exclude=exclude, title="CD", operations=avg)

CL = DataItem(data="surface_forces/totalCL", exclude=exclude, title="CL", operations=avg)

CDA = DataItem(
    data="surface_forces",
    exclude=exclude,
    title="CD*area",
    variables=[Variable(name="area", data="params.reference_geometry.area")],
    operations=[Expression(expr="totalCD * area"), avg],
)

CLf = DataItem(
    data="surface_forces",
    exclude=exclude,
    title="CLf",
    operations=[Expression(expr="1/2*totalCL + totalCMy"), avg],
)

CLr = DataItem(
    data="surface_forces",
    exclude=exclude,
    title="CLr",
    operations=[Expression(expr="1/2*totalCL - totalCMy"), avg],
)

CFy = DataItem(data="surface_forces/totalCFy", exclude=exclude, title="CS", operations=avg)

statistical_data = [
    "params/reference_geometry/area",
    CD,
    CDA,
    Delta(data=CD),
    CL,
    CLf,
    CLr,
    CFy,
    "volume_mesh/stats/n_nodes",
    "params/time_stepping/max_steps",
]
statistical_table = Table(
    data=statistical_data,
    section_title="Statistical data",
    formatter=[
        (
            None
            if d
            in [
                "params/reference_geometry/area",
                "volume_mesh/stats/n_nodes",
                "params/time_stepping/max_steps",
            ]
            else ".4f"
        )
        for d in statistical_data
    ],
)

geometry_screenshots = [
    Chart3D(
        section_title="Geometry",
        items_in_row=2,
        force_new_page=True,
        show="boundaries",
        camera=camera,
        exclude=exclude,
        fig_name=f"geo_{i}",
    )
    for i, camera in enumerate(cameras_geo)
]
cpt_screenshots = [
    Chart3D(
        section_title="Isosurface, Cpt=-1",
        items_in_row=2,
        force_new_page=True,
        show="isosurface",
        iso_field="Cpt",
        exclude=exclude,
        camera=camera,
    )
    for camera in cameras_cp
]
cfvec_screenshots = [
    Chart3D(
        section_title="CfVec",
        items_in_row=2,
        force_new_page=True,
        show="boundaries",
        field="CfVec",
        mode="lic",
        limits=(1e-4, 10),
        is_log_scale=True,
        exclude=exclude,
        camera=camera,
    )
    for camera in cameras_cp
]
y_slices_lic_screenshots = [
    Chart3D(
        section_title=f"Slice velocity LIC y={y}",
        items_in_row=2,
        force_new_page=True,
        show="slices",
        include=[f"slice_y_{name}"],
        field="velocityVec",
        mode="lic",
        limits=(0 * u.m / u.s, 50 * u.m / u.s),
        camera=side_camera_slice,
        fig_name=f"slice_y_vec_{name}",
    )
    for name, y in zip(["0", "0_2", "0_4", "0_6", "0_8"], [0, 0.2, 0.4, 0.6, 0.8])
]
y_slices_screenshots = [
    Chart3D(
        section_title=f"Slice velocity y={y}",
        items_in_row=2,
        force_new_page=True,
        show="slices",
        include=[f"slice_y_{name}"],
        field="velocity",
        limits=(0 * u.m / u.s, 50 * u.m / u.s),
        camera=side_camera_slice,
        fig_name=f"slice_y_{name}",
    )
    for name, y in zip(["0", "0_2", "0_4", "0_6", "0_8"], [0, 0.2, 0.4, 0.6, 0.8])
]
y_slices_lic_screenshots = [
    Chart3D(
        section_title=f"Slice velocity LIC y={y}",
        items_in_row=2,
        force_new_page=True,
        show="slices",
        include=[f"slice_y_{name}"],
        field="velocityVec",
        mode="lic",
        limits=(0 * u.m / u.s, 50 * u.m / u.s),
        camera=side_camera_slice,
        fig_name=f"slice_y_vec_{name}",
    )
    for name, y in zip(["0", "0_2", "0_4", "0_6", "0_8"], [0, 0.2, 0.4, 0.6, 0.8])
]
z_slices_screenshots = [
    Chart3D(
        section_title=f"Slice velocity z={z}",
        items_in_row=2,
        force_new_page=True,
        show="slices",
        include=[f"slice_z_{name}"],
        field="velocity",
        limits=(0 * u.m / u.s, 50 * u.m / u.s),
        camera=top_camera_slice,
        fig_name=f"slice_z_{name}",
    )
    for name, z in zip(["neg0_2", "0", "0_2", "0_4", "0_6", "0_8"], [-0.2, 0, 0.2, 0.4, 0.6, 0.8])
]
y_plus_screenshots = [
    Chart3D(
        section_title="y+",
        items_in_row=2,
        show="boundaries",
        field="yPlus",
        exclude=exclude,
        limits=(0, 5),
        camera=camera,
        fig_name=f"yplus_{i}",
    )
    for i, camera in enumerate([top_camera, bottom_camera])
]
cp_screenshots = [
    Chart3D(
        section_title="Cp",
        items_in_row=2,
        show="boundaries",
        field="Cp",
        exclude=exclude,
        limits=limits,
        camera=camera,
        fig_name=f"cp_{i}",
    )
    for i, (limits, camera) in enumerate(zip(limits_cp, cameras_cp))
]
cpx_screenshots = [
    Chart3D(
        section_title="Cpx",
        items_in_row=2,
        show="boundaries",
        field="Cpx",
        exclude=exclude,
        limits=(-0.3, 0.3),
        camera=camera,
        fig_name=f"cpx_{i}",
    )
    for i, camera in enumerate(cameras_cp)
]
wall_shear_screenshots = [
    Chart3D(
        section_title="Wall shear stress magnitude",
        items_in_row=2,
        show="boundaries",
        field="wallShearMag",
        exclude=exclude,
        limits=(0 * u.Pa, 5 * u.Pa),
        camera=camera,
        fig_name=f"wallShearMag_{i}",
    )
    for i, camera in enumerate(cameras_cp)
]

report = ReportTemplate(
    title="Aerodynamic analysis of DrivAer",
    items=[
        Summary(),
        Inputs(),
        statistical_table,
        Chart2D(
            x="x_slicing_force_distribution/X",
            y="x_slicing_force_distribution/totalCumulative_CD_Curve",
            fig_name="totalCumulative_CD_Curve",
            background="geometry",
            exclude=exclude,
        ),
        Chart2D(
            x="surface_forces/pseudo_step",
            y="surface_forces/totalCD",
            section_title="Drag Coefficient",
            fig_name="cd_fig",
            exclude=exclude,
            focus_x=(1 / 3, 1),
        ),
        *geometry_screenshots,
        *cp_screenshots,
        *cpx_screenshots,
        *cpt_screenshots,
        *y_slices_screenshots,
        *y_slices_lic_screenshots,
        *z_slices_screenshots,
        *y_plus_screenshots,
        *wall_shear_screenshots,
    ],
    settings=Settings(dpi=150),
)

report = report.create_in_cloud(
    f"{size}-{len(cases)}cases-slices-using-groups-Cpt, Cpx, wallShear, dpi=default",
    cases,
    solver_version=__solver_version__,
)

report.wait()
report.download("report.pdf")
