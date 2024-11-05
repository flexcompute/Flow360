import argparse
import json
import sys
import textwrap

parser = argparse.ArgumentParser(
    description="This program converts a given json file with BETDisk config into an updated version",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("input", type=str, help="input json file name")
parser.add_argument("output", type=str, help="output file name")
args = parser.parse_args()
if args.input.endswith(".json") != True:
    sys.exit("incorrect input")


# loading a config json file containing BETDisk inputs
with open(args.input) as reader:
    read = json.load(reader)
    read = json.dumps(read, indent=4)

    _, _, var3 = read.partition('"BETDisks": ')
    read = var3.replace(": ", "=")
    read = read.replace('"rotationDirectionRule"', "rotation_direction_rule")
    read = read.replace('"omega"', "omega")
    read = read.replace('"numberOfBlades"', "number_of_blades")
    read = read.replace('"twists"', "twists")
    read = read.replace('"chords"', "chords")
    read = read.replace('"ReynoldsNumbers"', "reynolds_numbers")
    read = read.replace('"chordRef"', "chord_ref")
    read = read.replace('"nLoadingNodes"', "n_loading_nodes")
    read = read.replace('"sectionalRadiuses"', "sectional_radiuses")
    read = read.replace('"sectionalPolars"', "sectional_polars")
    read = read.replace('"MachNumbers"', "mach_numbers")
    read = read.replace('"alphas"', "alphas")
    read = read.replace('"radius"', "radius")
    read = read.replace('"twist"', "twist")
    read = read.replace('"chord"', "chord")
    read = read.replace('"liftCoeffs"', "lift_coeffs")
    read = read.replace('"dragCoeffs"', "drag_coeffs")
    read = read.replace('"tipGap"', "tip_gap")
    read = read.replace('"bladeLineChord"', "blade_line_chord")
    read = read.replace('"initialBladeDirection"', "initial_blade_direction")

    # removing inputs that are no longer used
    v1, _, v3 = read.partition('"axisOfRotation"=[')
    _, _, rest = v3.partition("],\n            ")
    read = v1 + rest

    # removing inputs that are no longer used
    v1, _, v3 = read.partition('"centerOfRotation"=[')
    _, _, rest = v3.partition("],\n            ")
    read = v1 + rest

    # removing inputs that are no longer used
    v1, _, v3 = read.partition("\n            radius=")
    _, _, rest = v3.partition(",")
    read = v1 + rest

    # removing inputs that are no longer used
    v1, _, v3 = read.partition('"thickness"=')
    _, _, rest = v3.partition(",\n            ")
    read = v1 + rest

    # partition on twists
    twists1, twists2, twists3 = read.partition("twists=[")
    inner_twists1, inner_twists2, inner_twists3 = twists3.partition("]")
    inner_twists1 = inner_twists1.replace("{", "BETDiskTwist(").replace("}", ")")
    twists3 = inner_twists1 + inner_twists2 + inner_twists3
    read = twists1 + twists2 + twists3

    # partition on chords
    chords1, chords2, chords3 = read.partition("chords=[")
    inner_chords1, inner_chords2, inner_chords3 = chords3.partition("]")
    inner_chords1 = inner_chords1.replace("{", "BETDiskChord(").replace("}", ")")
    chords3 = inner_chords1 + inner_chords2 + inner_chords3
    read = chords1 + chords2 + chords3

    # partition on sectional_polars
    polars1, polars2, polars3 = read.partition("sectional_polars=[")
    inner_polars1, inner_polars2, inner_polars3 = polars3.partition(
        "]\n                        ]\n                    ]\n                }\n            ]"
    )
    inner_polars1 = inner_polars1.replace("{", "BETDiskSectionalPolar(").replace("}", ")")
    inner_polars2 = inner_polars2.replace("}", ")")
    polars3 = inner_polars1 + inner_polars2
    read = polars1 + polars2 + polars3

    # removing leftover brackets
    read = read.replace("[\n        {\n", "")

    # fixing indentation
    read = textwrap.indent(read, "    ")

# writing the converted input into a file
with open(args.output, "w") as file:
    file.write(read)
