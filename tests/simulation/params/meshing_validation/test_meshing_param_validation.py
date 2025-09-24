import pydantic as pd
import pytest

from flow360.component.simulation.meshing_param.params import MeshingParams
from flow360.component.simulation.meshing_param.volume_params import (
    AutomatedFarfield,
    AxisymmetricRefinement,
    RotationCylinder,
    UniformRefinement,
)
from flow360.component.simulation.primitives import AxisymmetricBody, Cylinder, Surface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import CGS_unit_system


def test_disable_invalid_axisymmetric_body_construction():
    with pytest.raises(
        pd.ValidationError,
        match=r"Expect profile samples to be (Axial, Radial) samples with positive Radial. Found invalid point: [-1.  1.  3.] cm.",
    ):
        with CGS_unit_system:
            cylinder_1 = AxisymmetricBody(
                name="1",
                axis=(0, 0, 1),
                center=(0, 5, 0),
                profile_curve=[(-1, 0), (-1, 1, 3), (1, 1), (1, 0)],
            )

    with pytest.raises(
        pd.ValidationError,
        match="Expect first profile sample to be (Axial, 0.0). Found invalid point: [-1.  1.] cm.",
    ):
        with CGS_unit_system:
            cylinder_1 = AxisymmetricBody(
                name="1",
                axis=(0, 0, 1),
                center=(0, 5, 0),
                profile_curve=[(-1, 1), (1, 2)],
            )

    with pytest.raises(
        pd.ValidationError,
        match="Expect profile samples to be (Axial, Radial) samples with positive Radial. Found invalid point: [-1.  1.  3.] cm.",
    ):
        with CGS_unit_system:
            cylinder_1 = AxisymmetricBody(
                name="1",
                axis=(0, 0, 1),
                center=(0, 5, 0),
                profile_curve=[(-1, 0), (-1, 1), (1, 1)],
            )


#
# def test_disable_multiple_cylinder_in_one_ratataion_cylinder():
#     with pytest.raises(
#         pd.ValidationError,
#         match="Only single instance is allowed in entities for each RotationCylinder.",
#     ):
#         with CGS_unit_system:
#             cylinder_1 = Cylinder(
#                 name="1",
#                 outer_radius=12,
#                 height=2,
#                 axis=(0, 1, 0),
#                 center=(0, 5, 0),
#             )
#             cylinder_2 = Cylinder(
#                 name="2",
#                 outer_radius=2,
#                 height=2,
#                 axis=(0, 1, 0),
#                 center=(0, 5, 0),
#             )
#             SimulationParams(
#                 meshing=MeshingParams(
#                     volume_zones=[
#                         RotationCylinder(
#                             entities=[cylinder_1, cylinder_2],
#                             spacing_axial=20,
#                             spacing_radial=0.2,
#                             spacing_circumferential=20,
#                             enclosed_entities=[
#                                 Surface(name="hub"),
#                             ],
#                         ),
#                         AutomatedFarfield(),
#                     ],
#                 )
#             )
#
#
# def test_limit_cylinder_entity_name_length_in_rotation_cylinder():
#     with pytest.raises(
#         pd.ValidationError,
#         match=r"The name \(very_long_cylinder_name\) of `Cylinder` entity in `RotationCylinder`"
#         + " exceeds 18 characters limit.",
#     ):
#         with CGS_unit_system:
#             cylinder = Cylinder(
#                 name="very_long_cylinder_name",
#                 outer_radius=12,
#                 height=2,
#                 axis=(0, 1, 0),
#                 center=(0, 5, 0),
#             )
#             SimulationParams(
#                 meshing=MeshingParams(
#                     volume_zones=[
#                         RotationCylinder(
#                             entities=[cylinder],
#                             spacing_axial=20,
#                             spacing_radial=0.2,
#                             spacing_circumferential=20,
#                             enclosed_entities=[
#                                 Surface(name="hub"),
#                             ],
#                         ),
#                         AutomatedFarfield(),
#                     ],
#                 )
#             )
#
#
# def test_reuse_of_same_cylinder():
#     with pytest.raises(
#         pd.ValidationError,
#         match=r"Using Volume entity `I am reused` in `AxisymmetricRefinement`, `RotationCylinder` at the same time is not allowed.",
#     ):
#         with CGS_unit_system:
#             cylinder = Cylinder(
#                 name="I am reused",
#                 outer_radius=1,
#                 height=12,
#                 axis=(0, 1, 0),
#                 center=(0, 5, 0),
#             )
#             SimulationParams(
#                 meshing=MeshingParams(
#                     volume_zones=[
#                         RotationCylinder(
#                             entities=[cylinder],
#                             spacing_axial=20,
#                             spacing_radial=0.2,
#                             spacing_circumferential=20,
#                             enclosed_entities=[
#                                 Surface(name="hub"),
#                             ],
#                         ),
#                         AutomatedFarfield(),
#                     ],
#                     refinements=[
#                         AxisymmetricRefinement(
#                             entities=[cylinder],
#                             spacing_axial=0.1,
#                             spacing_radial=0.2,
#                             spacing_circumferential=0.3,
#                         )
#                     ],
#                 )
#             )
#
#     with CGS_unit_system:
#         cylinder = Cylinder(
#             name="Okay to reuse",
#             outer_radius=1,
#             height=12,
#             axis=(0, 1, 0),
#             center=(0, 5, 0),
#         )
#         SimulationParams(
#             meshing=MeshingParams(
#                 volume_zones=[
#                     RotationCylinder(
#                         entities=[cylinder],
#                         spacing_axial=20,
#                         spacing_radial=0.2,
#                         spacing_circumferential=20,
#                         enclosed_entities=[
#                             Surface(name="hub"),
#                         ],
#                     ),
#                     AutomatedFarfield(),
#                 ],
#                 refinements=[
#                     UniformRefinement(
#                         entities=[cylinder],
#                         spacing=0.1,
#                     )
#                 ],
#             )
#         )
#
#     with pytest.raises(
#         pd.ValidationError,
#         match=r"Using Volume entity `I am reused` in `AxisymmetricRefinement`, `UniformRefinement` at the same time is not allowed.",
#     ):
#         with CGS_unit_system:
#             cylinder = Cylinder(
#                 name="I am reused",
#                 outer_radius=1,
#                 height=12,
#                 axis=(0, 1, 0),
#                 center=(0, 5, 0),
#             )
#             SimulationParams(
#                 meshing=MeshingParams(
#                     refinements=[
#                         UniformRefinement(entities=[cylinder], spacing=0.1),
#                         AxisymmetricRefinement(
#                             entities=[cylinder],
#                             spacing_axial=0.1,
#                             spacing_radial=0.1,
#                             spacing_circumferential=0.1,
#                         ),
#                     ],
#                 )
#             )
#
#     with pytest.raises(
#         pd.ValidationError,
#         match=r" Volume entity `I am reused` is used multiple times in `UniformRefinement`.",
#     ):
#         with CGS_unit_system:
#             cylinder = Cylinder(
#                 name="I am reused",
#                 outer_radius=1,
#                 height=12,
#                 axis=(0, 1, 0),
#                 center=(0, 5, 0),
#             )
#             SimulationParams(
#                 meshing=MeshingParams(
#                     refinements=[
#                         UniformRefinement(entities=[cylinder], spacing=0.1),
#                         UniformRefinement(entities=[cylinder], spacing=0.2),
#                     ],
#                 )
#             )
#
#
# def test_limit_cylinder_entity_name_length_in_rotation_cylinder():
#     with pytest.raises(
#         pd.ValidationError,
#         match=r"The name \(very_long_cylinder_name\) of `Cylinder` entity in `RotationCylinder`"
#         + " exceeds 18 characters limit.",
#     ):
#         with CGS_unit_system:
#             cylinder = Cylinder(
#                 name="very_long_cylinder_name",
#                 outer_radius=12,
#                 height=2,
#                 axis=(0, 1, 0),
#                 center=(0, 5, 0),
#             )
#             SimulationParams(
#                 meshing=MeshingParams(
#                     volume_zones=[
#                         RotationCylinder(
#                             entities=[cylinder],
#                             spacing_axial=20,
#                             spacing_radial=0.2,
#                             spacing_circumferential=20,
#                             enclosed_entities=[
#                                 Surface(name="hub"),
#                             ],
#                         ),
#                         AutomatedFarfield(),
#                     ],
#                 )
#             )
#
#
# def test_reuse_of_same_cylinder():
#     with pytest.raises(
#         pd.ValidationError,
#         match=r"Using Volume entity `I am reused` in `AxisymmetricRefinement`, `RotationCylinder` at the same time is not allowed.",
#     ):
#         with CGS_unit_system:
#             cylinder = Cylinder(
#                 name="I am reused",
#                 outer_radius=1,
#                 height=12,
#                 axis=(0, 1, 0),
#                 center=(0, 5, 0),
#             )
#             SimulationParams(
#                 meshing=MeshingParams(
#                     volume_zones=[
#                         RotationCylinder(
#                             entities=[cylinder],
#                             spacing_axial=20,
#                             spacing_radial=0.2,
#                             spacing_circumferential=20,
#                             enclosed_entities=[
#                                 Surface(name="hub"),
#                             ],
#                         ),
#                         AutomatedFarfield(),
#                     ],
#                     refinements=[
#                         AxisymmetricRefinement(
#                             entities=[cylinder],
#                             spacing_axial=0.1,
#                             spacing_radial=0.2,
#                             spacing_circumferential=0.3,
#                         )
#                     ],
#                 )
#             )
#
#     with CGS_unit_system:
#         cylinder = Cylinder(
#             name="Okay to reuse",
#             outer_radius=1,
#             height=12,
#             axis=(0, 1, 0),
#             center=(0, 5, 0),
#         )
#         SimulationParams(
#             meshing=MeshingParams(
#                 volume_zones=[
#                     RotationCylinder(
#                         entities=[cylinder],
#                         spacing_axial=20,
#                         spacing_radial=0.2,
#                         spacing_circumferential=20,
#                         enclosed_entities=[
#                             Surface(name="hub"),
#                         ],
#                     ),
#                     AutomatedFarfield(),
#                 ],
#                 refinements=[
#                     UniformRefinement(
#                         entities=[cylinder],
#                         spacing=0.1,
#                     )
#                 ],
#             )
#         )
#
#     with pytest.raises(
#         pd.ValidationError,
#         match=r"Using Volume entity `I am reused` in `AxisymmetricRefinement`, `UniformRefinement` at the same time is not allowed.",
#     ):
#         with CGS_unit_system:
#             cylinder = Cylinder(
#                 name="I am reused",
#                 outer_radius=1,
#                 height=12,
#                 axis=(0, 1, 0),
#                 center=(0, 5, 0),
#             )
#             SimulationParams(
#                 meshing=MeshingParams(
#                     refinements=[
#                         UniformRefinement(entities=[cylinder], spacing=0.1),
#                         AxisymmetricRefinement(
#                             entities=[cylinder],
#                             spacing_axial=0.1,
#                             spacing_radial=0.1,
#                             spacing_circumferential=0.1,
#                         ),
#                     ],
#                 )
#             )
#
#     with pytest.raises(
#         pd.ValidationError,
#         match=r" Volume entity `I am reused` is used multiple times in `UniformRefinement`.",
#     ):
#         with CGS_unit_system:
#             cylinder = Cylinder(
#                 name="I am reused",
#                 outer_radius=1,
#                 height=12,
#                 axis=(0, 1, 0),
#                 center=(0, 5, 0),
#             )
#             SimulationParams(
#                 meshing=MeshingParams(
#                     refinements=[
#                         UniformRefinement(entities=[cylinder], spacing=0.1),
#                         UniformRefinement(entities=[cylinder], spacing=0.2),
#                     ],
#                 )
#             )
