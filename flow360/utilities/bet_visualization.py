"""
BET disk plotting helpers for dictionaries produced by any translator
(CHARM, XROTOR, DFDC, XFOIL, C81).

These live on the client side because matplotlib is a client-only dependency
(too heavy for the schema package). Each plotting function only *builds and
returns* the matplotlib figures - it does not show or save them. The caller
owns all presentation: tweak axes/limits/styles, then ``fig.savefig(...)`` or
``plt.show()`` as preferred.
"""

import matplotlib.pyplot as plt

__all__ = ["plot_bet_polars", "plot_bet_geometry"]

_TICKS = ["x-", "+--", "p:", "^-", ">--", "<-", "*:", "-", "1-", "2--", "3--", "4:"]
_FIG_SIZE = (16, 8)


def _as_floats(values):
    """Strip unyt units (if present) and return a list of plain floats."""
    return [float(v.value) if hasattr(v, "value") else float(v) for v in values]


def _style_axes(axis, title, xlabel, ylabel):
    """Apply the shared grid/legend/labels styling to one axis."""
    axis.grid(True)
    axis.legend()
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)


def _station_polar_figures(polars, alphas, mach_nums, station_idx):
    """Build the two polar figures (Cl-Cd/Cl-alpha and Cl-alpha/Cd-alpha) for one station."""
    fig1, (ax_clcd, ax_clalpha) = plt.subplots(1, 2, figsize=_FIG_SIZE)
    fig2, (ax_cl, ax_cd) = plt.subplots(1, 2, figsize=_FIG_SIZE)
    for mach_idx, mach in enumerate(mach_nums):
        cl = polars["lift_coeffs"][mach_idx][0]
        cd = polars["drag_coeffs"][mach_idx][0]
        tick = _TICKS[mach_idx % len(_TICKS)]
        ax_clcd.plot(cd, cl, tick, label=f"Mach#:{mach:.2f}")
        ax_clalpha.plot(alphas, cl, tick, label=f"Mach#:{mach:.2f}")
        ax_cl.plot(alphas, cl, tick, label=f"Cl Mach#:{mach:.2f}")
        ax_cd.plot(alphas, cd, tick, label=f"Cd Mach#:{mach:.2f}")
    _style_axes(ax_clcd, f"Cl vs Cd (station {station_idx})", "Cd", "Cl")
    _style_axes(ax_clalpha, f"Cl vs Alpha (station {station_idx})", "Alpha", "Cl")
    _style_axes(ax_cl, f"Cl vs Alpha (station {station_idx})", "Alpha", "Cl")
    _style_axes(ax_cd, f"Cd vs Alpha (station {station_idx})", "Alpha", "Cd")
    return [fig1, fig2]


def plot_bet_polars(bet_disk):
    """
    Build CL/CD polar figures for a BET disk dictionary.

    For each radial station two figures are created:
      - CL vs CD and CL vs Alpha
      - CL vs Alpha and CD vs Alpha

    Parameters
    ----------
    bet_disk: dict
        BET disk dictionary (output of any generate_*_bet_json function).

    Returns
    -------
    list of matplotlib.figure.Figure, in station order (2 figures per station).
    """
    mach_nums = bet_disk["mach_numbers"]
    alphas = _as_floats(bet_disk["alphas"])

    figures = []
    for station_idx, polars in enumerate(bet_disk["sectional_polars"]):
        figures.extend(_station_polar_figures(polars, alphas, mach_nums, station_idx))
    return figures


def plot_bet_geometry(bet_disk):
    """
    Build twist- and chord-vs-radius figures for a BET disk dictionary.

    Parameters
    ----------
    bet_disk: dict
        BET disk dictionary (output of any generate_*_bet_json function).

    Returns
    -------
    list of matplotlib.figure.Figure: ``[twist_vs_radius, chord_vs_radius]``.
    """
    twist_radii = _as_floats(entry["radius"] for entry in bet_disk["twists"])
    twists = _as_floats(entry["twist"] for entry in bet_disk["twists"])
    chord_radii = _as_floats(entry["radius"] for entry in bet_disk["chords"])
    chords = _as_floats(entry["chord"] for entry in bet_disk["chords"])

    fig_twist, ax_twist = plt.subplots(figsize=_FIG_SIZE)
    ax_twist.plot(twist_radii, twists, "o-")
    ax_twist.set_xlabel("Radius")
    ax_twist.set_ylabel("Twist (deg)")
    ax_twist.set_title("Twist vs Propeller Radius")
    ax_twist.grid(True)

    fig_chord, ax_chord = plt.subplots(figsize=_FIG_SIZE)
    ax_chord.plot(chord_radii, chords, "o-")
    ax_chord.set_xlabel("Radius")
    ax_chord.set_ylabel("Chord")
    ax_chord.set_title("Chord vs Propeller Radius")
    ax_chord.grid(True)

    return [fig_twist, fig_chord]
