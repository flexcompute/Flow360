"""Client-side utility helpers.

Currently provides BET disk plotting helpers (see :mod:`bet_visualization`).
New utilities can be added as additional submodules here and re-exported below.

Usage::

    from flow360.utilities import plot_bet_polars, plot_bet_geometry
"""

from flow360.utilities.bet_visualization import plot_bet_geometry, plot_bet_polars

__all__ = ["plot_bet_polars", "plot_bet_geometry"]
