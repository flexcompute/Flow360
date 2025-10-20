import json
import os
import unittest

import pytest

import flow360 as fl
from flow360.component.simulation.framework.updater_utils import compare_values

assertions = unittest.TestCase("__init__")


def generate_BET_param(type, given_path_prefix: str = None):

    with fl.SI_unit_system:
        bet_cylinder_SI = fl.Cylinder(
            name="BET_cylinder", center=[0, 0, 0], axis=[0, 0, 1], outer_radius=3.81, height=15
        )

    with fl.imperial_unit_system:
        bet_cylinder_imperial = fl.Cylinder(
            name="BET_cylinder", center=[0, 0, 0], axis=[0, 0, 1], outer_radius=150, height=15
        )
    prepending_path = (
        given_path_prefix if given_path_prefix else os.path.dirname(os.path.abspath(__file__))
    )
    if type == "c81":
        param = fl.BETDisk.from_c81(
            file=fl.C81File(
                file_path=(os.path.join(prepending_path, "data/c81", "Xv15_geometry.csv"))
            ),
            rotation_direction_rule="leftHand",
            omega=0.0046 * fl.u.deg / fl.u.s,
            chord_ref=14 * fl.u.m,
            n_loading_nodes=20,
            entities=bet_cylinder_imperial,
            angle_unit=fl.u.deg,
            length_unit=fl.u.m,
            number_of_blades=3,
        )

    elif type == "dfdc":
        param = fl.BETDisk.from_dfdc(
            file=fl.DFDCFile(
                file_path=(os.path.join(prepending_path, "data", "dfdc_xv15_twist0.case"))
            ),
            rotation_direction_rule="leftHand",
            omega=0.0046 * fl.u.deg / fl.u.s,
            chord_ref=14 * fl.u.m,
            n_loading_nodes=20,
            entities=bet_cylinder_SI,
            angle_unit=fl.u.deg,
            length_unit=fl.u.m,
        )

    elif type == "xfoil":
        param = fl.BETDisk.from_xfoil(
            file=fl.XFOILFile(
                file_path=(
                    os.path.join(
                        prepending_path,
                        "data/xfoil",
                        "xv15_geometry_xfoil_translatorDisk0.csv",
                    )
                )
            ),
            rotation_direction_rule="leftHand",
            initial_blade_direction=[1, 0, 0],
            blade_line_chord=1 * fl.u.m,
            omega=0.0046 * fl.u.deg / fl.u.s,
            chord_ref=14 * fl.u.m,
            n_loading_nodes=20,
            entities=bet_cylinder_imperial,
            angle_unit=fl.u.deg,
            length_unit=fl.u.m,
            number_of_blades=3,
        )

    elif type == "xrotor":
        param = fl.BETDisk.from_xrotor(
            file=fl.XROTORFile(
                file_path=(
                    os.path.join(
                        prepending_path,
                        "data",
                        "xv15_like_twist0.xrotor",
                    )
                )
            ),
            rotation_direction_rule="leftHand",
            omega=0.0046 * fl.u.deg / fl.u.s,
            chord_ref=14 * fl.u.m,
            n_loading_nodes=20,
            entities=bet_cylinder_SI,
            angle_unit=fl.u.deg,
            length_unit=fl.u.m,
        )

    return param


def translate_and_compare(type, ref_json_file: str, atol=1e-15, rtol=1e-10, debug=False):
    translated = generate_BET_param(type)
    translated = translated.model_dump_json(
        exclude={
            "type_name",
            "private_attribute_constructor",
            "private_attribute_input_cache",
        }
    )
    translated = json.loads(translated)
    del translated["entities"]["stored_entities"][0]["private_attribute_id"]
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref", ref_json_file)) as fh:
        ref_dict = json.load(fh)
    if debug:
        print(">>> translated = ", translated)
        print("=== translated ===\n", json.dumps(translated, indent=4, sort_keys=True))
        print("=== ref_dict ===\n", json.dumps(ref_dict, indent=4, sort_keys=True))
    assert compare_values(
        ref_dict, translated, atol=atol, rtol=rtol, ignore_keys=["private_attribute_id"]
    )


def test_translated_c81_params():
    translate_and_compare(type="c81", ref_json_file="ref_c81.json")


def test_translated_dfdc_params():
    translate_and_compare(type="dfdc", ref_json_file="ref_dfdc.json")


def test_translated_xfoil_params():
    translate_and_compare(type="xfoil", ref_json_file="ref_xfoil.json")


def test_translated_xrotor_params():
    translate_and_compare(type="xrotor", ref_json_file="ref_xrotor.json")


def test_xrotor_params():

    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref", "xrotorTest.json")
    ) as fh:
        refbetFlow360 = json.load(fh)

    # Create BETDisk from xrotor file
    bet = generate_BET_param("xrotor")

    # Compare omega
    assertions.assertEqual(refbetFlow360["omega"], bet.omega)

    # Compare chord ref
    assertions.assertEqual(refbetFlow360["chordRef"], bet.chord_ref)

    # Compare n loading nodes
    assertions.assertEqual(refbetFlow360["nLoadingNodes"], bet.n_loading_nodes)

    # Compare number of blades
    assertions.assertEqual(refbetFlow360["numberOfBlades"], bet.number_of_blades)

    # Compare rotation direction rule
    assertions.assertEqual(refbetFlow360["rotationDirectionRule"], bet.rotation_direction_rule)

    # Compare twists
    for number, twist in enumerate(bet.twists):
        assertions.assertEqual(refbetFlow360["twists"][number]["radius"], twist.radius)
        assertions.assertEqual(refbetFlow360["twists"][number]["twist"], twist.twist)

    # Compare chords
    for number, chord in enumerate(bet.chords):
        assertions.assertEqual(refbetFlow360["chords"][number]["radius"], chord.radius)
        assertions.assertEqual(refbetFlow360["chords"][number]["chord"], chord.chord)

    # Compare alphas
    for number, alpha in enumerate(bet.alphas):
        assertions.assertEqual(refbetFlow360["alphas"][number], alpha)

    # Compare mach numbers
    for number, mach in enumerate(bet.mach_numbers):
        assertions.assertEqual(refbetFlow360["MachNumbers"][number], mach)

    # Compare reynold numbers
    for number, reynolds in enumerate(bet.reynolds_numbers):
        assertions.assertEqual(refbetFlow360["ReynoldsNumbers"][number], reynolds)

    # Compare sectional polars
    for number, polar in enumerate(bet.sectional_polars):

        # Lift coeffs
        for number1, lift_coeff_matrix_1 in enumerate(polar.lift_coeffs):
            for number2, lift_coeff_matrix_2 in enumerate(lift_coeff_matrix_1):
                for number3, lift_coeff_matrix_3 in enumerate(lift_coeff_matrix_2):
                    assertions.assertAlmostEqual(
                        refbetFlow360["sectionalPolars"][number]["liftCoeffs"][number1][number2][
                            number3
                        ],
                        lift_coeff_matrix_3,
                    )

        # Drag coeffs
        for number1, drag_coeff_matrix_1 in enumerate(polar.drag_coeffs):
            for number2, drag_coeff_matrix_2 in enumerate(drag_coeff_matrix_1):
                for number3, drag_coeff_matrix_3 in enumerate(drag_coeff_matrix_2):
                    assertions.assertAlmostEqual(
                        refbetFlow360["sectionalPolars"][number]["dragCoeffs"][number1][number2][
                            number3
                        ],
                        drag_coeff_matrix_3,
                    )

    # Compare sectional radiuses
    for number, radius in enumerate(bet.sectional_radiuses):
        assertions.assertEqual(refbetFlow360["sectionalRadiuses"][number], radius)

    # Compare cylinder inputs
    assertions.assertEqual(
        refbetFlow360["axisOfRotation"], list(bet.entities.stored_entities[0].axis)
    )
    assertions.assertEqual(
        refbetFlow360["centerOfRotation"], list(bet.entities.stored_entities[0].center)
    )
    assertions.assertEqual(refbetFlow360["thickness"], bet.entities.stored_entities[0].height)
    assertions.assertEqual(refbetFlow360["radius"], bet.entities.stored_entities[0].outer_radius)


def test_dfdc_params():

    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref", "dfdcTest.json")
    ) as fh:
        refbetFlow360 = json.load(fh)

    # Create BETDisk from xrotor file
    bet = generate_BET_param("dfdc")

    # Compare omega
    assertions.assertEqual(refbetFlow360["omega"], bet.omega)

    # Compare chord ref
    assertions.assertEqual(refbetFlow360["chordRef"], bet.chord_ref)

    # Compare n loading nodes
    assertions.assertEqual(refbetFlow360["nLoadingNodes"], bet.n_loading_nodes)

    # Compare number of blades
    assertions.assertEqual(refbetFlow360["numberOfBlades"], bet.number_of_blades)

    # Compare rotation direction rule
    assertions.assertEqual(refbetFlow360["rotationDirectionRule"], bet.rotation_direction_rule)

    # Compare twists
    for number, twist in enumerate(bet.twists):
        assertions.assertEqual(refbetFlow360["twists"][number]["radius"], twist.radius)
        assertions.assertEqual(refbetFlow360["twists"][number]["twist"], twist.twist)

    # Compare chords
    for number, chord in enumerate(bet.chords):
        assertions.assertEqual(refbetFlow360["chords"][number]["radius"], chord.radius)
        assertions.assertEqual(refbetFlow360["chords"][number]["chord"], chord.chord)

    # Compare alphas
    for number, alpha in enumerate(bet.alphas):
        assertions.assertEqual(refbetFlow360["alphas"][number], alpha)

    # Compare mach numbers
    for number, mach in enumerate(bet.mach_numbers):
        assertions.assertEqual(refbetFlow360["MachNumbers"][number], mach)

    # Compare reynold numbers
    for number, reynolds in enumerate(bet.reynolds_numbers):
        assertions.assertEqual(refbetFlow360["ReynoldsNumbers"][number], reynolds)

    # Compare sectional polars
    for number, polar in enumerate(bet.sectional_polars):

        # Lift coeffs
        for number1, lift_coeff_matrix_1 in enumerate(polar.lift_coeffs):
            for number2, lift_coeff_matrix_2 in enumerate(lift_coeff_matrix_1):
                for number3, lift_coeff_matrix_3 in enumerate(lift_coeff_matrix_2):
                    assertions.assertAlmostEqual(
                        refbetFlow360["sectionalPolars"][number]["liftCoeffs"][number1][number2][
                            number3
                        ],
                        lift_coeff_matrix_3,
                    )

        # Drag coeffs
        for number1, drag_coeff_matrix_1 in enumerate(polar.drag_coeffs):
            for number2, drag_coeff_matrix_2 in enumerate(drag_coeff_matrix_1):
                for number3, drag_coeff_matrix_3 in enumerate(drag_coeff_matrix_2):
                    assertions.assertAlmostEqual(
                        refbetFlow360["sectionalPolars"][number]["dragCoeffs"][number1][number2][
                            number3
                        ],
                        drag_coeff_matrix_3,
                    )

    # Compare sectional radiuses
    for number, radius in enumerate(bet.sectional_radiuses):
        assertions.assertEqual(refbetFlow360["sectionalRadiuses"][number], radius)

    # Compare cylinder inputs
    assertions.assertEqual(
        refbetFlow360["axisOfRotation"], list(bet.entities.stored_entities[0].axis)
    )
    assertions.assertEqual(
        refbetFlow360["centerOfRotation"], list(bet.entities.stored_entities[0].center)
    )
    assertions.assertEqual(refbetFlow360["thickness"], bet.entities.stored_entities[0].height)
    assertions.assertEqual(refbetFlow360["radius"], bet.entities.stored_entities[0].outer_radius)


def test_c81_params():

    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref", "c81Test.json")
    ) as fh:
        refbetFlow360 = json.load(fh)

    # Create BETDisk from xrotor file
    bet = generate_BET_param("c81")

    # Compare omega
    assertions.assertEqual(refbetFlow360["omega"], bet.omega)

    # Compare chord ref
    assertions.assertEqual(refbetFlow360["chordRef"], bet.chord_ref)

    # Compare n loading nodes
    assertions.assertEqual(refbetFlow360["nLoadingNodes"], bet.n_loading_nodes)

    # Compare number of blades
    assertions.assertEqual(refbetFlow360["numberOfBlades"], bet.number_of_blades)

    # Compare rotation direction rule
    assertions.assertEqual(refbetFlow360["rotationDirectionRule"], bet.rotation_direction_rule)

    # Compare twists
    for number, twist in enumerate(bet.twists):
        assertions.assertEqual(refbetFlow360["twists"][number]["radius"], twist.radius)
        assertions.assertEqual(refbetFlow360["twists"][number]["twist"], twist.twist)

    # Compare chords
    for number, chord in enumerate(bet.chords):
        assertions.assertEqual(refbetFlow360["chords"][number]["radius"], chord.radius)
        assertions.assertEqual(refbetFlow360["chords"][number]["chord"], chord.chord)

    # Compare alphas
    for number, alpha in enumerate(bet.alphas):
        assertions.assertEqual(refbetFlow360["alphas"][number], alpha)

    # Compare mach numbers
    for number, mach in enumerate(bet.mach_numbers):
        assertions.assertEqual(refbetFlow360["MachNumbers"][number], mach)

    # Compare reynold numbers
    for number, reynolds in enumerate(bet.reynolds_numbers):
        assertions.assertEqual(refbetFlow360["ReynoldsNumbers"][number], reynolds)

    # Compare sectional polars
    for number, polar in enumerate(bet.sectional_polars):

        # Lift coeffs
        for number1, lift_coeff_matrix_1 in enumerate(polar.lift_coeffs):
            for number2, lift_coeff_matrix_2 in enumerate(lift_coeff_matrix_1):
                for number3, lift_coeff_matrix_3 in enumerate(lift_coeff_matrix_2):
                    assertions.assertAlmostEqual(
                        refbetFlow360["sectionalPolars"][number]["liftCoeffs"][number1][number2][
                            number3
                        ],
                        lift_coeff_matrix_3,
                    )

        # Drag coeffs
        for number1, drag_coeff_matrix_1 in enumerate(polar.drag_coeffs):
            for number2, drag_coeff_matrix_2 in enumerate(drag_coeff_matrix_1):
                for number3, drag_coeff_matrix_3 in enumerate(drag_coeff_matrix_2):
                    assertions.assertAlmostEqual(
                        refbetFlow360["sectionalPolars"][number]["dragCoeffs"][number1][number2][
                            number3
                        ],
                        drag_coeff_matrix_3,
                    )

    # Compare sectional radiuses
    for number, radius in enumerate(bet.sectional_radiuses):
        assertions.assertEqual(refbetFlow360["sectionalRadiuses"][number], radius)

    # Compare cylinder inputs
    assertions.assertEqual(
        refbetFlow360["axisOfRotation"], list(bet.entities.stored_entities[0].axis)
    )
    assertions.assertEqual(
        refbetFlow360["centerOfRotation"], list(bet.entities.stored_entities[0].center)
    )
    assertions.assertEqual(refbetFlow360["thickness"], bet.entities.stored_entities[0].height)
    assertions.assertEqual(refbetFlow360["radius"], bet.entities.stored_entities[0].outer_radius)


def test_xfoil_params():

    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref", "xfoilTest.json")
    ) as fh:
        refbetFlow360 = json.load(fh)

    # Create BETDisk from xrotor file
    bet = generate_BET_param("xfoil")

    # Compare omega
    assertions.assertEqual(refbetFlow360["omega"], bet.omega)

    # Compare chord ref
    assertions.assertEqual(refbetFlow360["chordRef"], bet.chord_ref)

    # Compare n loading nodes
    assertions.assertEqual(refbetFlow360["nLoadingNodes"], bet.n_loading_nodes)

    # Compare number of blades
    assertions.assertEqual(refbetFlow360["numberOfBlades"], bet.number_of_blades)

    # Compare rotation direction rule
    assertions.assertEqual(refbetFlow360["rotationDirectionRule"], bet.rotation_direction_rule)

    # Compare twists
    for number, twist in enumerate(bet.twists):
        assertions.assertEqual(refbetFlow360["twists"][number]["radius"], twist.radius)
        assertions.assertEqual(refbetFlow360["twists"][number]["twist"], twist.twist)

    # Compare chords
    for number, chord in enumerate(bet.chords):
        assertions.assertEqual(refbetFlow360["chords"][number]["radius"], chord.radius)
        assertions.assertEqual(refbetFlow360["chords"][number]["chord"], chord.chord)

    # Compare alphas
    for number, alpha in enumerate(bet.alphas):
        assertions.assertEqual(refbetFlow360["alphas"][number], alpha)

    # Compare mach numbers
    for number, mach in enumerate(bet.mach_numbers):
        assertions.assertEqual(refbetFlow360["MachNumbers"][number], mach)

    # Compare reynold numbers
    for number, reynolds in enumerate(bet.reynolds_numbers):
        assertions.assertEqual(refbetFlow360["ReynoldsNumbers"][number], reynolds)

    # Compare sectional polars
    for number, polar in enumerate(bet.sectional_polars):

        # Lift coeffs
        for number1, lift_coeff_matrix_1 in enumerate(polar.lift_coeffs):
            for number2, lift_coeff_matrix_2 in enumerate(lift_coeff_matrix_1):
                for number3, lift_coeff_matrix_3 in enumerate(lift_coeff_matrix_2):
                    assertions.assertAlmostEqual(
                        refbetFlow360["sectionalPolars"][number]["liftCoeffs"][number1][number2][
                            number3
                        ],
                        lift_coeff_matrix_3,
                    )

        # Drag coeffs
        for number1, drag_coeff_matrix_1 in enumerate(polar.drag_coeffs):
            for number2, drag_coeff_matrix_2 in enumerate(drag_coeff_matrix_1):
                for number3, drag_coeff_matrix_3 in enumerate(drag_coeff_matrix_2):
                    assertions.assertAlmostEqual(
                        refbetFlow360["sectionalPolars"][number]["dragCoeffs"][number1][number2][
                            number3
                        ],
                        drag_coeff_matrix_3,
                    )

    # Compare sectional radiuses
    for number, radius in enumerate(bet.sectional_radiuses):
        assertions.assertEqual(refbetFlow360["sectionalRadiuses"][number], radius)

    # Compare cylinder inputs
    assertions.assertEqual(
        refbetFlow360["axisOfRotation"], list(bet.entities.stored_entities[0].axis)
    )
    assertions.assertEqual(
        refbetFlow360["centerOfRotation"], list(bet.entities.stored_entities[0].center)
    )
    assertions.assertEqual(refbetFlow360["thickness"], bet.entities.stored_entities[0].height)
    assertions.assertEqual(refbetFlow360["radius"], bet.entities.stored_entities[0].outer_radius)


def test_file_model():
    """
    Test the C81File model's construction, immutability, and serialization.

    This test verifies:
    1. Normal object creation with valid file path
       - Instantiates a C81File object using a valid CSV file path from test data

    2. Immutable file_path enforcement
       - Ensures the 'file_path' field cannot be modified post-creation by attempting
         assignment and expecting a ValueError with frozen fields message

    3. Model serialization/deserialization round-trip
       - Validates that dumping the model to a dictionary and re-initializing produces
         an equivalent object using Pydantic's model_dump method

    The test ensures proper behavior for both normal usage scenarios and edge cases
    like attempting field modification.
    """
    # 1: Normal construction
    file = fl.C81File(
        file_path=(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "data/c81", "Xv15_geometry.csv"
            )
        )
    )

    # 2: possible use case. Change file_path using assignment will be prohibited.
    with pytest.raises(ValueError, match="Cannot modify immutable/frozen fields: file_path"):
        file.file_path = "non_existing_file.csv"

    # 3. Deserialization
    kwargs = file.model_dump()
    file_2 = fl.C81File(**kwargs)
