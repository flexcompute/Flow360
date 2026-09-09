"""Translator for CHARM BET input files."""

import logging
import os
from math import cos, radians, sin, sqrt
from typing import Literal

import numpy as np
import pydantic as pd
import unyt as u

from flow360_schema.exceptions import Flow360ValueError
from flow360_schema.framework.entity.entity_list import EntityList
from flow360_schema.framework.entity.geometric_types import Axis
from flow360_schema.framework.physical_dimensions import Angle, Length
from flow360_schema.framework.physical_dimensions import AngularVelocity as AngularVelocityDim
from flow360_schema.models.entities.volume_entities import Cylinder
from flow360_schema.models.simulation.models.bet.bet_translator_interface import (
    blend_polars_to_flat_plate,
    get_file_content,
    set_up_bet_dict_with_user_inputs,
)

logger = logging.getLogger(__name__)


def _charm_int(token):
    """
    Parse a CHARM integer field, tolerating Fortran free-format decimals
    (e.g. ``"3"`` or ``"3.0"`` -> ``3``). CHARM integer fields (NBLADE,
    IROTAT, ITILT, NROTOR, NSEC, repeat counts) are occasionally written
    with a trailing decimal point, which a bare ``int()`` would reject.
    """
    return int(float(token))


def _expand_charm_repeat_notation(tokens):
    """
    Expand CHARM repeat notation (e.g., '20*0.0', '16*0.1', '12*-3.0', '11*0')
    into a flat list of floats.

    Attributes
    ----------
    tokens: list of str, whitespace-split tokens from a CHARM input file line
    return: list of float
    """
    result = []
    for token in tokens:
        if "*" in token:
            parts = token.split("*")
            count = _charm_int(parts[0])
            value = float(parts[1])
            result.extend([value] * count)
        else:
            result.append(float(token))
    return result


_CHARM_BG_KNOWN_KEYWORDS = {
    "KBGEOM",
    "NSEG",
    "CUTOUT",
    "SL",
    "CHORD",
    "ELOFSG",
    "SWEEPD",
    "TWRD",
    "TWSTGD",
    "ANHD",
    "THCKND",
    "THIKND",
    "KFLAP",
    "FLAPND",
    "FLHNGE",
    "FLDEFL",
    "NCAM",
    "NCHORD",
}

_CHARM_BG_SCALAR_KEYWORDS = {"KBGEOM", "NSEG", "NCAM", "TWRD"}
_CHARM_BG_ARRAY_KEYWORDS = {
    "SL",
    "CHORD",
    "ELOFSG",
    "SWEEPD",
    "TWSTGD",
    "ANHD",
    "THCKND",
    "THIKND",
    "KFLAP",
    "FLAPND",
    "FLHNGE",
    "FLDEFL",
}


def _charm_bg_keyword(line_text):
    """
    Determine if a line in a CHARM BG file is a keyword line.

    CHARM BG keywords are uppercase identifiers optionally followed by parenthetical
    suffixes like (ISEG), (ISEG+1), or descriptive text. Examples:
      'NSEG', 'SL(ISEG)', 'THIKND(ISEG+1)', 'TWRD (Blade root twist ...)',
      'NCHORD  NSPAN  ICOS'

    Attributes
    ----------
    line_text: str, stripped line from a BG file
    return: str or None, the base keyword if this is a keyword line, else None
    """
    if not line_text:
        return None
    first_token = line_text.split()[0].upper()
    base = first_token.split("(")[0]
    if base in _CHARM_BG_KNOWN_KEYWORDS:
        return base
    return None


def parse_charm_blade_geometry(bg_file_content):
    """
    Parse a CHARM blade geometry (.bg) file.

    Handles format variations across CHARM versions:
      - Keywords with suffixes: SL(ISEG), THIKND(ISEG+1), TWRD (description...)
      - Both THCKND and THIKND spellings for thickness
      - CUTOUT with 1 or 3 values (takes first as radial cutout)
      - Repeat notation: 20*0.0, 16*0.1, 11*0
      - First line may be a description (skipped automatically)

    Attributes
    ----------
    bg_file_content: str, content of the CHARM blade geometry file
    return: dict with parsed blade geometry data including:
            nseg, cutout, sl, chord, twrd, twstgd, and optionally thcknd, sweepd, anhd
    """
    lines = bg_file_content.strip().split("\n")

    bg_dict = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        kw = _charm_bg_keyword(line)
        if kw is None:
            i += 1
            continue

        if kw == "NCHORD":
            break  # End of blade data section

        # Normalize THIKND -> thcknd
        store_key = "thcknd" if kw == "THIKND" else kw.lower()

        if kw == "CUTOUT":
            # CUTOUT: next line has 1 or 3 values; take first as radial cutout
            i += 1
            tokens = lines[i].strip().split()
            bg_dict["cutout"] = float(tokens[0])
            i += 1
            continue

        if kw in _CHARM_BG_SCALAR_KEYWORDS:
            i += 1
            value_line = lines[i].strip()
            tokens = value_line.split()
            bg_dict[store_key] = float(tokens[0]) if "." in tokens[0] else int(tokens[0])
            i += 1
            continue

        if kw in _CHARM_BG_ARRAY_KEYWORDS:
            values = []
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                if not next_line:
                    i += 1
                    continue
                if _charm_bg_keyword(next_line) is not None:
                    break
                tokens = next_line.split()
                values.extend(_expand_charm_repeat_notation(tokens))
                i += 1
            bg_dict[store_key] = values
            continue

        i += 1

    # Validate required fields
    for key in ["nseg", "cutout", "sl", "chord", "twrd", "twstgd"]:
        if key not in bg_dict:
            raise Flow360ValueError(f"CHARM blade geometry file missing required field: {key.upper()}")

    nseg = int(bg_dict["nseg"])
    if len(bg_dict["sl"]) != nseg:
        raise Flow360ValueError(f"Expected {nseg} SL values, got {len(bg_dict['sl'])}")
    # TWSTGD is indexed twstgd[i] for i in range(nseg) downstream, so it
    # must contain at least NSEG entries — otherwise we'd hit IndexError or
    # silently use the wrong twist distribution.
    twstgd = bg_dict.get("twstgd", [])
    if len(twstgd) < nseg:
        raise Flow360ValueError(f"Expected at least {nseg} TWSTGD values, got {len(twstgd)}")
    # CHORD must have at least one value: translation pads a short chord list by
    # repeating its last entry, which is impossible (IndexError) from an empty
    # list, so reject a present-but-empty CHORD with a clear error here.
    if len(bg_dict.get("chord", [])) == 0:
        raise Flow360ValueError("CHARM blade geometry file has CHORD with no numeric values")

    return bg_dict


def parse_charm_airfoil_tables(af_file_content):
    """
    Parse a CHARM airfoil tables file containing multi-section CL/CD data.

    Handles format variations across CHARM versions:
      - Optional ALLOW_TO_EXCEED prefix line
      - Data rows that wrap across multiple lines (e.g., 11 Mach columns)
      - Comment lines as 'COMMENT#N' or descriptive text + separator
      - Position-based dimension string parsing (handles embedded spaces)
      - Header with 2 or 3 values (NSEC ICMPR [flag])

    Attributes
    ----------
    af_file_content: str, content of the CHARM airfoil tables file
    return: dict with nsec, radial_positions, thickness_ratios, sections[]
            Each section has: mach_numbers, alphas, cl[mach][alpha], cd[mach][alpha]
    """
    lines = af_file_content.strip().split("\n")
    idx = 0

    # Skip optional ALLOW_TO_EXCEED line
    if lines[idx].strip().startswith("ALLOW_TO_EXCEED"):
        idx += 1

    # Line: NSEC ICMPR [flag]
    header = lines[idx].split()
    nsec = _charm_int(header[0])
    idx += 1

    # Radial positions
    radial_positions = [float(x) for x in lines[idx].split()]
    idx += 1
    # One radial position per section: translation interpolates section polars
    # against these radii (np.interp needs len(radial_positions) == nsec), so a
    # mismatched count must fail here with a clear error rather than a later
    # numpy ValueError.
    if len(radial_positions) != nsec:
        raise Flow360ValueError(
            f"CHARM airfoil tables file lists {len(radial_positions)} radial positions "
            f"but NSEC={nsec}; the counts must match (one radial position per section)."
        )

    # Thickness ratios
    thickness_ratios = [float(x) for x in lines[idx].split()]
    idx += 1
    # One thickness ratio per section, same as radial positions.
    if len(thickness_ratios) != nsec:
        raise Flow360ValueError(
            f"CHARM airfoil tables file lists {len(thickness_ratios)} thickness ratios "
            f"but NSEC={nsec}; the counts must match (one thickness ratio per section)."
        )

    def _read_n_floats(start, count):
        """Read exactly `count` float values, spanning as many lines as needed."""
        vals = []
        cur = start
        while len(vals) < count and cur < len(lines):
            for t in lines[cur].strip().split():
                vals.append(float(t))
                if len(vals) == count:
                    break
            cur += 1
        if len(vals) < count:
            raise Flow360ValueError(
                f"CHARM airfoil tables file is truncated: expected {count} values "
                f"but only {len(vals)} were found before end of file."
            )
        return vals, cur

    def _read_table_row(start, ncols):
        """Read one data row: alpha + ncols coefficients, possibly multi-line."""
        vals = []
        needed = ncols + 1
        cur = start
        while len(vals) < needed and cur < len(lines):
            for t in lines[cur].strip().split():
                vals.append(float(t))
                if len(vals) == needed:
                    break
            cur += 1
        if len(vals) < needed:
            raise Flow360ValueError(
                f"CHARM airfoil tables file is truncated: expected a data row of {needed} values "
                f"(alpha + {ncols} coefficients) but only {len(vals)} were found before end of file."
            )
        return vals[0], vals[1 : ncols + 1], cur

    sections = []
    for _ in range(nsec):
        # Skip 2 comment/separator lines
        idx += 1
        idx += 1

        # Airfoil name + dimension string (position-based at column 30)
        dim_line = lines[idx]
        if len(dim_line) >= 42:
            dim_str = dim_line[30:42]
        else:
            # Fallback: extract trailing digits/spaces
            dim_part = ""
            for ch in reversed(dim_line.rstrip()):
                if ch.isdigit() or ch == " ":
                    dim_part = ch + dim_part
                else:
                    break
            dim_str = dim_part.rjust(12)

        nlmach = int(dim_str[0:2])
        nlaoa = int(dim_str[2:4])
        ndmach = int(dim_str[4:6])
        ndaoa = int(dim_str[6:8])
        nmmach = int(dim_str[8:10])
        nmaoa = int(dim_str[10:12])
        idx += 1

        # CL: Mach number line(s) then nlaoa data rows
        cl_machs, idx = _read_n_floats(idx, nlmach)
        cl_alphas = []
        cl_data = np.zeros((nlaoa, nlmach))
        for row in range(nlaoa):
            alpha, coeffs, idx = _read_table_row(idx, nlmach)
            cl_alphas.append(alpha)
            cl_data[row, :] = coeffs

        # CD: Mach number line(s) then ndaoa data rows
        cd_machs, idx = _read_n_floats(idx, ndmach)
        cd_alphas = []
        cd_data = np.zeros((ndaoa, ndmach))
        for row in range(ndaoa):
            alpha, coeffs, idx = _read_table_row(idx, ndmach)
            cd_alphas.append(alpha)
            cd_data[row, :] = coeffs

        # CM: Mach number line(s) then nmaoa data rows (advance idx, don't store)
        _cm_machs, idx = _read_n_floats(idx, nmmach)
        for _ in range(nmaoa):
            _a, _c, idx = _read_table_row(idx, nmmach)

        # Transpose to mach-major: [alpha][mach] -> [mach][alpha]
        cl_data = cl_data.T
        cd_data = cd_data.T

        # Unify CD onto the CL alpha grid so a single blend covers both polars
        if cd_alphas != cl_alphas:
            cd_unified = np.zeros((cd_data.shape[0], len(cl_alphas)))
            for m_idx in range(cd_data.shape[0]):
                cd_unified[m_idx, :] = np.interp(cl_alphas, cd_alphas, cd_data[m_idx, :])
            cd_data = cd_unified
            cd_alphas = list(cl_alphas)

        # Reconcile CD onto the CL Mach grid. CHARM permits a separate CD Mach
        # count/grid (ndmach, cd_machs) from the CL grid (nlmach, cl_machs), but
        # everything downstream pairs CL and CD by Mach index, so interpolate CD
        # along the Mach axis onto cl_machs. Without this, a section whose CD Mach
        # grid differs from its CL grid raises an IndexError during flat-plate
        # blending or silently mismatches drag to the wrong Mach number.
        if cd_machs != cl_machs:
            cd_on_cl_mach = np.zeros((len(cl_machs), cd_data.shape[1]))
            cd_machs_arr = np.array(cd_machs)
            for a_idx in range(cd_data.shape[1]):
                cd_on_cl_mach[:, a_idx] = np.interp(cl_machs, cd_machs_arr, cd_data[:, a_idx])
            cd_data = cd_on_cl_mach
            cd_machs = list(cl_machs)

        # If the CL alpha range does not cover -180..180, blend to flat-plate
        if cl_alphas[0] != -180 or cl_alphas[-1] != 180:
            mach_keys = [str(m) for m in cl_machs]
            cl_dict = {k: list(cl_data[i, :]) for i, k in enumerate(mach_keys)}
            cd_dict = {k: list(cd_data[i, :]) for i, k in enumerate(mach_keys)}
            cl_alphas, _, cl_dict, cd_dict = blend_polars_to_flat_plate(cl_alphas, mach_keys, cl_dict, cd_dict)
            cd_alphas = list(cl_alphas)
            cl_data = np.array([cl_dict[k] for k in mach_keys])
            cd_data = np.array([cd_dict[k] for k in mach_keys])

        sections.append(
            {
                "mach_numbers": cl_machs,
                "alphas": cl_alphas,
                "cl": cl_data.tolist(),
                "cd_machs": cd_machs,
                "cd_alphas": cd_alphas,
                "cd": cd_data.tolist(),
            }
        )

    return {
        "nsec": nsec,
        "radial_positions": radial_positions,
        "thickness_ratios": thickness_ratios,
        "sections": sections,
    }


def parse_charm_rotor_wake(rw_file_content):
    """
    Parse a CHARM rotor wake (.rw) file for NBLADE, OMEGA, IROTAT, XROTOR, XTILT.

    The IROTAT line in a CHARM .rw file contains, on a single data row:
        IROTAT  XROTOR(1,2,3)  XTILT(1,2,3)  ITILT
    i.e. one int, three floats (rotor hub x,y,z), three floats (tilt angles in
    degrees about x, y, z axes), and one int.

    Per Dan Wachspress's CHARM conventions, fixed-wing "rotors" are flagged two
    ways: the file may start with ``NRTOO`` (half-wing pair marker) and/or have
    ``NBLADE=1`` with ``OMEGA=0.0``. This parser detects both.

    Attributes
    ----------
    rw_file_content: str, content of the CHARM rotor wake file
    return: dict with nblade (int), omega (float, rad/s), irotat (int: 1 or -1),
            xrotor (tuple of 3 floats, rotor hub position x,y,z),
            xtilt  (tuple of 3 floats, tilt angles in degrees about x,y,z),
            itilt  (int, order-flag for the tilt rotations),
            nrtoo  (bool, True if the file starts with the NRTOO half-wing marker),
            is_fixed_wing (bool, True if nrtoo OR (nblade<=1 and omega<=0)).
    """
    lines = rw_file_content.strip().split("\n")
    rw_dict = {"nrtoo": False}

    i = 0
    while i < len(lines):
        line = lines[i].strip().upper()

        if line.startswith("NRTOO"):
            rw_dict["nrtoo"] = True

        elif line.startswith("NBLADE"):
            i += 1
            tokens = lines[i].strip().split()
            rw_dict["nblade"] = _charm_int(tokens[0])
            rw_dict["omega"] = float(tokens[1])

        elif line.startswith("IROTAT"):
            i += 1
            tokens = lines[i].strip().split()
            rw_dict["irotat"] = _charm_int(tokens[0])
            if len(tokens) >= 8:
                rw_dict["xrotor"] = (
                    float(tokens[1]),
                    float(tokens[2]),
                    float(tokens[3]),
                )
                rw_dict["xtilt"] = (
                    float(tokens[4]),
                    float(tokens[5]),
                    float(tokens[6]),
                )
                rw_dict["itilt"] = _charm_int(tokens[7])
            # Do not break here: NBLADE/OMEGA may appear after IROTAT in the
            # file. Field order is not guaranteed, so scan to the end to avoid
            # leaving nblade/omega at their defaults (which would misconfigure
            # blade count and rotation rate).

        i += 1

    nblade = rw_dict.get("nblade", 1)
    omega = rw_dict.get("omega", 0.0)
    rw_dict["is_fixed_wing"] = bool(rw_dict["nrtoo"] or (nblade <= 1 and omega <= 0.0))

    return rw_dict


# CHARM ITILT value for which the implemented Rz * Ry * Rx tilt sequence is correct.
# CHARM's ITILT flag selects the order in which the XTILT(1,2,3) rotations are
# composed; only the order below is implemented here. This is assumed to be the
# CHARM default (flag 0) -- confirm against the CHARM manual / Dan Wachspress's
# conventions and update if a different value denotes the x->y->z order.
CHARM_SUPPORTED_ITILT = 0


def charm_xtilt_to_axis(xtilt_deg, itilt=None, default_axis=(0.0, 0.0, 1.0)):
    """
    Convert CHARM XTILT angles (in degrees, about the x, y, z axes) into a
    unit rotor-axis direction vector by applying Rz * Ry * Rx to ``default_axis``.

    Only the Rz * Ry * Rx composition order (XTILT applied x, then y, then z) is
    implemented. CHARM's ITILT flag can request a different rotation order; when
    ``itilt`` is provided and is not :data:`CHARM_SUPPORTED_ITILT`, a warning is
    logged and the implemented order is used anyway, so callers are alerted that
    the resulting axis may not match CHARM's intended tilt sequence rather than
    silently receiving a wrong axis.

    Attributes
    ----------
    xtilt_deg: sequence of 3 floats, (tilt_x, tilt_y, tilt_z) in degrees
    itilt: optional int, CHARM tilt-order flag from the rw file. When None, no
           order check is performed. When set and not equal to
           :data:`CHARM_SUPPORTED_ITILT`, a warning is logged.
    default_axis: sequence of 3 floats, the rotor axis when all tilts are zero.
                  Defaults to (0, 0, 1) (rotor thrust along +z in the BET frame).
    return: tuple of 3 floats, the rotated axis direction (unit vector).
    """
    if itilt is not None and int(itilt) != CHARM_SUPPORTED_ITILT:
        logger.warning(
            "CHARM: ITILT=%s requests a tilt rotation order that is not implemented; "
            "applying the default Rz*Ry*Rx (x->y->z) order. The rotor axis may not "
            "match CHARM's intended tilt sequence.",
            itilt,
        )
    tx, ty, tz = (radians(float(a)) for a in xtilt_deg)
    cx, sx = cos(tx), sin(tx)
    cy, sy = cos(ty), sin(ty)
    cz, sz = cos(tz), sin(tz)
    ax, ay, az = default_axis

    # Rx
    x1, y1, z1 = ax, cx * ay - sx * az, sx * ay + cx * az
    # Ry
    x2, y2, z2 = cy * x1 + sy * z1, y1, -sy * x1 + cy * z1
    # Rz
    x3, y3, z3 = cz * x2 - sz * y2, sz * x2 + cz * y2, z2

    norm = sqrt(x3 * x3 + y3 * y3 + z3 * z3)
    if norm == 0.0:
        return (ax, ay, az)
    return (x3 / norm, y3 / norm, z3 / norm)


def parse_charm_master_input(master_file_content):
    """
    Parse a CHARM Run Characteristics (.inp) master file to discover per-rotor file sets.

    Each rotor has 5 files listed: rw, bg, bd, af, (optional/none).
    Also extracts NROTOR and PATHNAME.

    Attributes
    ----------
    master_file_content: str, content of the CHARM master input file
    return: dict with nrotor, pathname, and rotors[] (list of dicts with file keys:
            rw_file, bg_file, af_file)
    """
    lines = master_file_content.strip().split("\n")
    result = {"nrotor": 0, "pathname": "", "rotors": []}

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.upper().startswith("NROTOR"):
            i += 1
            tokens = lines[i].strip().split()
            result["nrotor"] = _charm_int(tokens[0])

        elif line.upper().startswith("PATHNAME"):
            i += 1
            result["pathname"] = lines[i].strip()

        elif line.upper().startswith("INPUT FILENAMES"):
            # The rotor file names follow the header: rw, bg, bd, af (and an
            # optional trailing "none"). af_file is at i+4, so at least that many
            # lines must exist; otherwise the master file is truncated.
            if i + 4 >= len(lines):
                raise Flow360ValueError(
                    "CHARM master file is truncated: 'INPUT FILENAMES' block is missing its "
                    "rotor file lines (expected rw/bg/bd/af on the following lines)."
                )
            rw_file = lines[i + 1].strip()
            bg_file = lines[i + 2].strip()
            # bd_file = lines[i + 3].strip()  # ignored
            af_file = lines[i + 4].strip()
            result["rotors"].append(
                {
                    "rw_file": rw_file,
                    "bg_file": bg_file,
                    "af_file": af_file,
                }
            )
            # Advance only past the four lines we consumed (rw/bg/bd/af). The
            # optional trailing "none" line is not a recognised keyword, so the
            # main scan loop skips it harmlessly; hard-coding a 5-line skip would
            # swallow the next rotor's header when that "none" line is omitted.
            i += 4

        elif line.upper().startswith("SSPD"):
            break  # Done with file listing section

        i += 1

    return result


def translate_charm_to_bet_dict(bg_file_content, af_file_content, length_unit, angle_unit):
    """
    Translate CHARM blade geometry and airfoil tables into a BET disk dictionary.

    Attributes
    ----------
    bg_file_content: str, content of the CHARM blade geometry file
    af_file_content: str, content of the CHARM airfoil tables file
    length_unit: Length.PositiveFloat64, unit for lengths
    angle_unit: Angle.PositiveFloat64, unit for angles
    return: dict, BET disk dictionary
    """
    bg = parse_charm_blade_geometry(bg_file_content)
    af = parse_charm_airfoil_tables(af_file_content)

    nseg = int(bg["nseg"])
    cutout = bg["cutout"]
    sl = bg["sl"]
    chord_vals = list(bg["chord"])
    twrd = bg["twrd"]
    twstgd = bg["twstgd"]

    # Compute radial positions (NSEG+1 stations from cutout to tip)
    n_stations = nseg + 1
    r_stations = np.zeros(n_stations)
    r_stations[0] = cutout
    for i in range(nseg):
        r_stations[i + 1] = r_stations[i] + sl[i]

    # Compute twist at each station (cumulative incremental)
    twist_vals = np.zeros(n_stations)
    twist_vals[0] = twrd
    for i in range(nseg):
        twist_vals[i + 1] = twist_vals[i] + twstgd[i]

    # Handle chord array length mismatch
    if len(chord_vals) < n_stations:
        logger.warning(f"CHARM: Expected {n_stations} chord values but got {len(chord_vals)}. Padding with last value.")
        while len(chord_vals) < n_stations:
            chord_vals.append(chord_vals[-1])
    elif len(chord_vals) > n_stations:
        chord_vals = chord_vals[:n_stations]

    # Build twist and chord arrays in BET format
    twists = []
    chords = []
    for i in range(n_stations):
        twists.append({"radius": r_stations[i] * length_unit, "twist": twist_vals[i] * angle_unit})
        chords.append({"radius": r_stations[i] * length_unit, "chord": chord_vals[i] * length_unit})

    # Get Mach numbers and alphas from the airfoil tables (first section as reference)
    mach_numbers = af["sections"][0]["mach_numbers"]
    cl_alphas_raw = np.array(af["sections"][0]["alphas"])
    n_machs = len(mach_numbers)
    reynolds_numbers = [1]

    # Use CL alpha grid as the common grid; interpolate CD onto it if they differ
    alphas_deg = cl_alphas_raw
    n_alphas = len(alphas_deg)

    # Airfoil section radial positions and their CL/CD data
    af_radii = np.array(af["radial_positions"])
    n_af_sections = af["nsec"]

    # Build 3D arrays on common alpha and Mach grids: af_cl/af_cd
    # [section][mach][alpha]. Different sections can have different alpha grids
    # (CHARM blending may extend one section to +/-180 while leaving another at
    # its native range) AND different Mach grids (CHARM allows each section its
    # own Mach tables). The BET disk uses a single shared alpha grid and Mach
    # list, so interpolate every section's CL/CD onto both common grids. Indexing
    # every section by section zero's Mach count instead would IndexError on
    # sections with fewer Mach tables and silently drop extra ones.
    common_machs = np.array(mach_numbers)
    af_cl = np.zeros((n_af_sections, n_machs, n_alphas))
    af_cd = np.zeros((n_af_sections, n_machs, n_alphas))
    for sec_idx in range(n_af_sections):
        sec = af["sections"][sec_idx]
        sec_cl_alphas = np.array(sec["alphas"])
        sec_cd_alphas = np.array(sec["cd_alphas"])
        sec_machs = np.array(sec["mach_numbers"])
        n_sec_machs = len(sec_machs)

        # First put this section's CL/CD on the common alpha grid, keeping the
        # section's own Mach rows.
        sec_cl = np.zeros((n_sec_machs, n_alphas))
        sec_cd = np.zeros((n_sec_machs, n_alphas))
        for m_idx in range(n_sec_machs):
            cl_raw = np.array(sec["cl"][m_idx])
            if len(sec_cl_alphas) != n_alphas or not np.allclose(sec_cl_alphas, alphas_deg):
                sec_cl[m_idx, :] = np.interp(alphas_deg, sec_cl_alphas, cl_raw)
            else:
                sec_cl[m_idx, :] = cl_raw

            # Interpolate CD onto common alpha grid if it differs
            cd_raw = np.array(sec["cd"][m_idx])
            if len(sec_cd_alphas) != n_alphas or not np.allclose(sec_cd_alphas, alphas_deg):
                sec_cd[m_idx, :] = np.interp(alphas_deg, sec_cd_alphas, cd_raw)
            else:
                sec_cd[m_idx, :] = cd_raw

        # Then reconcile this section's Mach grid onto the common Mach grid.
        if n_sec_machs == n_machs and np.allclose(sec_machs, common_machs):
            af_cl[sec_idx] = sec_cl
            af_cd[sec_idx] = sec_cd
        else:
            for a_idx in range(n_alphas):
                af_cl[sec_idx, :, a_idx] = np.interp(common_machs, sec_machs, sec_cl[:, a_idx])
                af_cd[sec_idx, :, a_idx] = np.interp(common_machs, sec_machs, sec_cd[:, a_idx])

    # Interpolate airfoil polars to each geometry station (clamps at boundaries)
    sectional_polars = []
    for station_idx in range(n_stations):
        r = r_stations[station_idx]
        secpol = {"lift_coeffs": [], "drag_coeffs": []}

        for m_idx in range(n_machs):
            cl_at_sections = af_cl[:, m_idx, :]
            cd_at_sections = af_cd[:, m_idx, :]

            cl_interp = np.zeros(n_alphas)
            cd_interp = np.zeros(n_alphas)
            for a_idx in range(n_alphas):
                cl_interp[a_idx] = np.interp(r, af_radii, cl_at_sections[:, a_idx])
                cd_interp[a_idx] = np.interp(r, af_radii, cd_at_sections[:, a_idx])

            secpol["lift_coeffs"].append([cl_interp.tolist()])
            secpol["drag_coeffs"].append([cd_interp.tolist()])

        sectional_polars.append(secpol)

    bet_disk = {
        "sectional_radiuses": [r * length_unit for r in r_stations],
        "twists": twists,
        "chords": chords,
        "mach_numbers": mach_numbers,
        "reynolds_numbers": reynolds_numbers,
        "alphas": [a * angle_unit for a in alphas_deg],
        "sectional_polars": sectional_polars,
    }

    return bet_disk


@pd.validate_call
def generate_charm_bet_json(
    blade_geometry_file_content: str,
    airfoil_tables_file_content: str,
    rotation_direction_rule: Literal["leftHand", "rightHand"],
    initial_blade_direction: Axis | None,
    blade_line_chord: Length.NonNegativeFloat64,
    omega: AngularVelocityDim.NonNegativeFloat64,
    chord_ref: Length.PositiveFloat64,
    n_loading_nodes: pd.StrictInt,
    entities: EntityList[Cylinder],
    angle_unit: Angle.PositiveFloat64,
    length_unit: Length.PositiveFloat64,
    number_of_blades: pd.StrictInt,
    name: str,
) -> dict:
    """
    Takes in CHARM blade geometry and airfoil tables files and translates
    them into a flow360 BET input dictionary.

    The blade geometry file (.bg) and airfoil tables file (.af/.inp) can be
    discovered automatically using parse_charm_master_input() on a CHARM Run
    Characteristics file, or supplied directly.

    ``entities`` accepts a plain list of Cylinders; @validate_call coerces it
    into the EntityList consumed downstream (EntityList has no public ctor).

    return: dict, BET disk dictionary
    """
    bet_disk = translate_charm_to_bet_dict(
        bg_file_content=blade_geometry_file_content,
        af_file_content=airfoil_tables_file_content,
        length_unit=length_unit,
        angle_unit=angle_unit,
    )
    bet_disk = set_up_bet_dict_with_user_inputs(
        bet_disk=bet_disk,
        name=name,
        entities=entities,
        omega=omega,
        chord_ref=chord_ref,
        n_loading_nodes=n_loading_nodes,
        rotation_direction_rule=rotation_direction_rule,
        initial_blade_direction=initial_blade_direction,
        blade_line_chord=blade_line_chord,
        number_of_blades=number_of_blades,
    )
    return bet_disk


@pd.validate_call
def generate_charm_bet_json_from_master(
    master_file_path: str,
    rotation_direction_rule: Literal["leftHand", "rightHand"],
    initial_blade_direction: Axis | None,
    blade_line_chord: Length.NonNegativeFloat64,
    omega: AngularVelocityDim.NonNegativeFloat64 | None,
    chord_ref: Length.PositiveFloat64,
    n_loading_nodes: pd.StrictInt,
    entities: EntityList[Cylinder],
    angle_unit: Angle.PositiveFloat64,
    length_unit: Length.PositiveFloat64,
    name: str,
    resolve_pathname: bool = False,
    include_fixed_wing_entries: bool = False,
) -> list:
    """
    Takes a CHARM Run Characteristics master input file and translates all rotors
    into flow360 BET input dictionaries.

    Automatically discovers the blade geometry (.bg), airfoil tables (.af), and
    rotor wake (.rw) files referenced in the master file. Extracts NBLADE and OMEGA
    from the rotor wake file for each rotor.

    Attributes
    ----------
    master_file_path: str, path to the CHARM master .inp file
    rotation_direction_rule: str, "leftHand" or "rightHand"
    initial_blade_direction: Axis or None
    blade_line_chord: Length.NonNegativeFloat64
    omega: AngularVelocity.NonNegativeFloat64 or None (if None, uses OMEGA from rw file)
    chord_ref: Length.PositiveFloat64
    n_loading_nodes: int
    entities: EntityList[Cylinder], one Cylinder per rotating rotor, in master
             file order. Each rotor gets its own cylinder (rotors have distinct
             XROTOR centers / XTILT axes); the count must equal the number of
             emitted rotors (fixed-wing entries skipped unless
             include_fixed_wing_entries is True), otherwise Flow360ValueError is
             raised. A single-rotor master with one Cylinder behaves as before.
    angle_unit: Angle.PositiveFloat64
    length_unit: Length.PositiveFloat64
    name: str, base name for disks (appended with rotor index)
    resolve_pathname: bool, if True resolve rotor file paths against
                     master_dir + master["pathname"] (CHARM convention). Default
                     False, assuming the rotor files live alongside the master file.
    include_fixed_wing_entries: bool, if False (default) skip CHARM "rotors" that
                     are actually fixed-wing components (detected via the NRTOO
                     marker or NBLADE<=1 with OMEGA<=0). BETDisk only models
                     rotating blades, so wings/tails produce spurious non-rotating
                     disks. Matches translate_charm_master_to_bet_bundle.
    return: list of dict, one BET disk dictionary per (rotating) rotor
    """
    master_content = get_file_content(master_file_path)
    master = parse_charm_master_input(master_content)
    # Normalize to an absolute path first (like translate_charm_master_to_bet_bundle)
    # so rotor file resolution is independent of the process working directory; a
    # relative master path would otherwise yield an empty/relative base_path.
    master_dir = os.path.dirname(os.path.abspath(master_file_path))
    # Match translate_charm_master_to_bet_bundle's path semantics (same default
    # resolve_pathname=False): only join the master's PATHNAME when explicitly
    # requested, otherwise assume the rotor files live alongside the master file.
    base_path = os.path.join(master_dir, master["pathname"]) if resolve_pathname else master_dir

    # Each rotor has its own XROTOR center / XTILT axis, so it needs its own
    # cylinder. Map entities to rotors by emitted-rotor index (one Cylinder per
    # rotor, in master order); see the count validation below.
    provided_entities = entities.stored_entities
    n_provided = len(provided_entities)

    bet_disks = []
    for rotor_idx, rotor_files in enumerate(master["rotors"]):
        bg_path = os.path.join(base_path, rotor_files["bg_file"])
        af_path = os.path.join(base_path, rotor_files["af_file"])
        rw_path = os.path.join(base_path, rotor_files["rw_file"])

        # Extract NBLADE, OMEGA, IROTAT from rotor wake file. Defaults match
        # translate_charm_master_to_bet_bundle (nblade=1). Parsed before reading
        # bg/af so fixed-wing entries can be skipped without the extra work.
        rw_content = get_file_content(rw_path)
        rw_data = parse_charm_rotor_wake(rw_content)

        # Skip CHARM fixed-wing entries (wings/tails) by default, matching
        # translate_charm_master_to_bet_bundle. BETDisk only models rotating
        # blades, so emitting a disk for these produces spurious non-rotating
        # disks in full-aircraft masters.
        if bool(rw_data.get("is_fixed_wing", False)) and not include_fixed_wing_entries:
            rotor_label = os.path.splitext(os.path.basename(bg_path))[0]
            logger.info(
                f"CHARM: skipping fixed-wing entry '{rotor_label}' "
                f"(NRTOO={rw_data.get('nrtoo', False)}, NBLADE={int(rw_data.get('nblade', 1))}, "
                f"OMEGA={float(rw_data.get('omega', 0.0))})."
            )
            continue

        bg_content = get_file_content(bg_path)
        af_content = get_file_content(af_path)

        # Clamp to at least 1, matching the bundle helpers' max(nblade, 1). A raw
        # NBLADE=0 would otherwise be dropped by set_up_bet_dict_with_user_inputs
        # (its `if number_of_blades:` guard), leaving the two CHARM entry points
        # disagreeing on blade count.
        number_of_blades = max(int(rw_data.get("nblade", 1)), 1)
        # When omega is not supplied, take it from the rw file. Match the bundle
        # helpers: a non-positive rw OMEGA means "not rotating" -> 0 rpm (rotation
        # direction is conveyed separately via IROTAT), not a 0 or negative rad/s.
        if omega is not None:
            rotor_omega = omega
        else:
            rw_omega_rad_s = float(rw_data.get("omega", 0.0))
            rotor_omega = rw_omega_rad_s * u.rad / u.s if rw_omega_rad_s > 0 else 0 * u.rpm

        # Honour IROTAT from the rw file, defaulting to 1 (right-hand) when the
        # file omits it — matching translate_charm_master_to_bet_bundle, which
        # uses rw_data.get("irotat", 1). CHARM convention: 1 = right-hand
        # (counter-clockwise from above), -1 = left-hand. Warn if the caller's
        # rule disagrees, then trust the rw file.
        irotat = int(rw_data.get("irotat", 1))
        rw_rule = "rightHand" if irotat == 1 else "leftHand"
        if rw_rule != rotation_direction_rule:
            logger.warning(
                "CHARM rotor %s: caller's rotation_direction_rule=%s "
                "disagrees with rw IROTAT=%s (=> %s). Using rw value.",
                rotor_idx,
                rotation_direction_rule,
                irotat,
                rw_rule,
            )
        effective_rotation_rule = rw_rule

        # One cylinder per rotor, indexed by emitted-rotor position (== number of
        # disks built so far, since fixed-wing entries are skipped above).
        emit_idx = len(bet_disks)
        if emit_idx >= n_provided:
            raise Flow360ValueError(
                f"CHARM master file has more rotating rotors than provided cylinder entities "
                f"({n_provided}). Supply exactly one Cylinder per rotating rotor, in master order."
            )

        # Suffix by the emitted-rotor index (emit_idx), not the raw master index:
        # skipped fixed-wing entries would otherwise leave gaps (e.g. _rotor0,
        # _rotor2) while the cylinder mapping uses the contiguous emitted index.
        # n_provided equals the emitted rotor count (validated below), so use it
        # to decide whether a suffix is needed.
        disk_name = f"{name}_rotor{emit_idx}" if n_provided > 1 else name

        bet_disk = translate_charm_to_bet_dict(
            bg_file_content=bg_content,
            af_file_content=af_content,
            length_unit=length_unit,
            angle_unit=angle_unit,
        )
        bet_disk = set_up_bet_dict_with_user_inputs(
            bet_disk=bet_disk,
            name=disk_name,
            entities=entities,
            omega=rotor_omega,
            chord_ref=chord_ref,
            n_loading_nodes=n_loading_nodes,
            rotation_direction_rule=effective_rotation_rule,
            initial_blade_direction=initial_blade_direction,
            blade_line_chord=blade_line_chord,
            number_of_blades=number_of_blades,
        )
        # Override the shared entity list set by set_up_bet_dict_with_user_inputs
        # with this rotor's single cylinder so each disk maps to its own region.
        bet_disk["entities"] = [provided_entities[emit_idx]]
        bet_disks.append(bet_disk)

    if len(bet_disks) != n_provided:
        raise Flow360ValueError(
            f"Provided {n_provided} cylinder entities but {len(bet_disks)} rotating rotor(s) "
            f"were emitted from the master file. Supply exactly one Cylinder per rotating rotor, "
            f"in master order (fixed-wing entries are skipped unless include_fixed_wing_entries=True)."
        )

    return bet_disks


def translate_charm_master_to_bet_bundle(
    master_file_path,
    length_unit: Length.PositiveFloat64,
    angle_unit: Angle.PositiveFloat64,
    resolve_pathname=False,
    include_fixed_wing_entries=False,
):
    """
    One-shot CHARM -> Flow360 BET bundle.

    Parses a CHARM Run Characteristics master .inp file and, for every rotor
    listed in it, returns everything a caller needs to construct an
    :class:`fl.Cylinder` and an :class:`fl.BETDisk`:

      - Reads the rw / bg / af files for each rotor (resolved against the
        master file's directory by default; set ``resolve_pathname=True`` to
        honour the master file's PATHNAME field instead).
      - Calls :func:`parse_charm_rotor_wake` to get NBLADE, OMEGA, IROTAT,
        XROTOR and XTILT.
      - Calls :func:`translate_charm_to_bet_dict` to build the BET-disk
        polar/geometry dictionary.
      - Computes the cylinder ``center`` (from XROTOR) and ``axis``
        (from :func:`charm_xtilt_to_axis` applied to XTILT).

    To visualize the result, pass each entry's ``bet_data`` to the client-side
    ``plot_bet_polars`` / ``plot_bet_geometry`` helpers.

    Attributes
    ----------
    master_file_path: str, path to the CHARM master .inp file
    length_unit: Length.PositiveFloat64, unit for lengths (e.g. fl.u.m)
    angle_unit: Angle.PositiveFloat64, unit for angles (e.g. fl.u.deg)
    resolve_pathname: bool, if True resolve rotor file paths against
                     ``master_dir + master["pathname"]`` (CHARM convention).
                     Default is False, which assumes the rotor files live
                     alongside the master file.
    include_fixed_wing_entries: bool, if False (default) skip CHARM "rotors"
                     that are actually fixed-wing components (detected via the
                     ``NRTOO`` marker or NBLADE<=1 with OMEGA<=0). BETDisk only
                     models rotating blades, so wings/tails are not useful.

    Returns
    -------
    list of dict, one entry per rotor in the master file. Each dict has keys:
      - ``rotor_label``: str, stem of the bg filename
      - ``nblade``: int
      - ``omega``: AngularVelocity quantity (rad/s from the rw file; 0 rpm for
                   fixed-wing entries)
      - ``irotat``: int, 1 (counter-clockwise viewed from above) or -1 (clockwise)
      - ``rotation_direction_rule``: str, "rightHand" if irotat==1 else "leftHand"
      - ``xrotor``: tuple of 3 floats, raw XROTOR values from the rw file
      - ``xtilt``:  tuple of 3 floats, raw XTILT values (degrees) from the rw file
      - ``center``: Length 3-vector quantity, ``xrotor * length_unit``
      - ``axis``:   tuple of 3 floats, unit vector from XTILT rotations of +z
      - ``tip_radius``: Length quantity, last sectional radius from ``bet_data``
      - ``bet_data``: dict from :func:`translate_charm_to_bet_dict` (ready to
                     splat as ``**bet_data`` into ``fl.BETDisk(...)``)
    """
    master_content = get_file_content(master_file_path)
    master = parse_charm_master_input(master_content)
    master_dir = os.path.dirname(os.path.abspath(master_file_path))
    base_path = os.path.join(master_dir, master["pathname"]) if resolve_pathname else master_dir

    bundles = []
    for rotor_files in master["rotors"]:
        bg_path = os.path.join(base_path, rotor_files["bg_file"])
        af_path = os.path.join(base_path, rotor_files["af_file"])
        rw_path = os.path.join(base_path, rotor_files["rw_file"])

        rw_data = parse_charm_rotor_wake(get_file_content(rw_path))
        nblade = int(rw_data.get("nblade", 1))
        omega_rad_s = float(rw_data.get("omega", 0.0))
        irotat = int(rw_data.get("irotat", 1))
        xrotor = rw_data.get("xrotor", (0.0, 0.0, 0.0))
        xtilt = rw_data.get("xtilt", (0.0, 0.0, 0.0))
        itilt = rw_data.get("itilt")
        is_fixed_wing = bool(rw_data.get("is_fixed_wing", False))

        rotor_label = os.path.splitext(os.path.basename(bg_path))[0]

        if is_fixed_wing and not include_fixed_wing_entries:
            logger.info(
                f"CHARM: skipping fixed-wing entry '{rotor_label}' "
                f"(NRTOO={rw_data.get('nrtoo', False)}, NBLADE={nblade}, OMEGA={omega_rad_s})."
            )
            continue

        bg_content = get_file_content(bg_path)
        bet_data = translate_charm_to_bet_dict(
            bg_file_content=bg_content,
            af_file_content=get_file_content(af_path),
            length_unit=length_unit,
            angle_unit=angle_unit,
        )
        bg_parsed = parse_charm_blade_geometry(bg_content)
        max_thcknd = max(bg_parsed.get("thcknd", [0.0]) or [0.0])
        # Derive chord stats from the translated disk geometry (bet_data), not the
        # raw BG CHORD list: translate_charm_to_bet_dict pads/truncates chords to
        # the station count, so raw stats would not match the emitted BET disk.
        disk_chords = [float(c["chord"].to_value(length_unit)) for c in bet_data["chords"]]
        max_chord = max(disk_chords)
        mean_chord = sum(disk_chords) / len(disk_chords)

        omega_qty = omega_rad_s * u.rad / u.s if omega_rad_s > 0 else 0 * u.rpm

        bundles.append(
            {
                "rotor_label": rotor_label,
                "nblade": max(nblade, 1),
                "omega": omega_qty,
                "irotat": irotat,
                "rotation_direction_rule": "rightHand" if irotat == 1 else "leftHand",
                "xrotor": tuple(xrotor),
                "xtilt": tuple(xtilt),
                "center": tuple(xrotor) * length_unit,
                "axis": charm_xtilt_to_axis(xtilt, itilt=itilt),
                "tip_radius": bet_data["sectional_radiuses"][-1],
                "max_thcknd": max_thcknd,
                "max_chord": max_chord,
                "mean_chord": mean_chord,
                "chord_ref": mean_chord * length_unit,
                "max_blade_thickness": max_thcknd * max_chord * length_unit,
                "bet_data": bet_data,
            }
        )

    return bundles


def translate_charm_rotor_to_bet_bundle(
    blade_geometry_file_path,
    airfoil_tables_file_path,
    length_unit: Length.PositiveFloat64,
    angle_unit: Angle.PositiveFloat64,
    rotor_wake_file_path=None,
    nblade=None,
    omega=None,
    center=None,
    axis=None,
    rotation_direction_rule=None,
    rotor_label=None,
):
    """
    One-shot CHARM -> Flow360 BET bundle for a single rotor (no master .inp).

    Same return shape as :func:`translate_charm_master_to_bet_bundle`, but for
    cases where the user only has a blade-geometry file and an airfoil-tables
    file (the CHARM ``vertical_aero_charm_model_flexcompute`` drop is one
    example: it ships ``prop0bg.inp`` + ``fwd_aerofoil_tables.inp`` but no rw
    or master file).

    Values are resolved in this order: caller overrides (``nblade``, ``omega``,
    ``center``, ``axis``, ``rotation_direction_rule``) take precedence. When an
    override is not supplied and ``rotor_wake_file_path`` is given, the value
    is taken from the rw file. Otherwise defaults are used (``nblade=1``,
    ``omega=0 rpm``, ``center=origin``, ``axis=+z``, ``rotation_direction_rule=rightHand``).

    Attributes
    ----------
    blade_geometry_file_path: str, path to the CHARM ``*bg.inp`` file
    airfoil_tables_file_path: str, path to the CHARM airfoil tables file
                              (e.g. ``*af.inp`` / ``0012.inp`` / ``fwd_aerofoil_tables.inp``)
    length_unit, angle_unit: see :func:`translate_charm_master_to_bet_bundle`
    rotor_wake_file_path: str or None, optional ``*rw.inp`` to source
                          NBLADE/OMEGA/XROTOR/XTILT
    nblade, omega, center, axis, rotation_direction_rule: optional caller
                          overrides; see module docstring for units
    rotor_label: str or None, label for the rotor. Defaults to the bg-file stem.

    Returns
    -------
    dict with the same keys as one entry of
    :func:`translate_charm_master_to_bet_bundle`. Pass ``bet_data`` to the
    client-side ``plot_bet_polars`` / ``plot_bet_geometry`` helpers to plot.
    """
    if rotor_label is None:
        rotor_label = os.path.splitext(os.path.basename(blade_geometry_file_path))[0]

    bg_content = get_file_content(blade_geometry_file_path)
    bet_data = translate_charm_to_bet_dict(
        bg_file_content=bg_content,
        af_file_content=get_file_content(airfoil_tables_file_path),
        length_unit=length_unit,
        angle_unit=angle_unit,
    )
    bg_parsed = parse_charm_blade_geometry(bg_content)
    max_thcknd = max(bg_parsed.get("thcknd", [0.0]) or [0.0])
    # Derive chord stats from the translated disk geometry (bet_data), not the raw
    # BG CHORD list: translate_charm_to_bet_dict pads/truncates chords to the
    # station count, so raw stats would not match the emitted BET disk.
    disk_chords = [float(c["chord"].to_value(length_unit)) for c in bet_data["chords"]]
    max_chord = max(disk_chords)
    mean_chord = sum(disk_chords) / len(disk_chords)

    rw_nblade, rw_omega_rad_s, rw_irotat = 1, 0.0, 1
    rw_xrotor, rw_xtilt = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    rw_itilt = None
    if rotor_wake_file_path is not None:
        rw_data = parse_charm_rotor_wake(get_file_content(rotor_wake_file_path))
        rw_nblade = int(rw_data.get("nblade", rw_nblade))
        rw_omega_rad_s = float(rw_data.get("omega", rw_omega_rad_s))
        rw_irotat = int(rw_data.get("irotat", rw_irotat))
        rw_xrotor = rw_data.get("xrotor", rw_xrotor)
        rw_xtilt = rw_data.get("xtilt", rw_xtilt)
        rw_itilt = rw_data.get("itilt")

    final_nblade = int(nblade) if nblade is not None else rw_nblade
    if omega is not None:
        final_omega = omega
    elif rw_omega_rad_s > 0:
        final_omega = rw_omega_rad_s * u.rad / u.s
    else:
        final_omega = 0 * u.rpm
    final_center = center if center is not None else tuple(rw_xrotor) * length_unit
    final_axis = axis if axis is not None else charm_xtilt_to_axis(rw_xtilt, itilt=rw_itilt)
    if rotation_direction_rule is None:
        rotation_direction_rule = "rightHand" if rw_irotat == 1 else "leftHand"

    return {
        "rotor_label": rotor_label,
        "nblade": max(final_nblade, 1),
        "omega": final_omega,
        "irotat": rw_irotat,
        "rotation_direction_rule": rotation_direction_rule,
        "xrotor": tuple(rw_xrotor),
        "xtilt": tuple(rw_xtilt),
        "center": final_center,
        "axis": final_axis,
        "tip_radius": bet_data["sectional_radiuses"][-1],
        "max_thcknd": max_thcknd,
        "max_chord": max_chord,
        "mean_chord": mean_chord,
        "chord_ref": mean_chord * length_unit,
        "max_blade_thickness": max_thcknd * max_chord * length_unit,
        "bet_data": bet_data,
    }
