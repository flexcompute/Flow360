"""
Shared helper functions for results processing tests.
"""

import numpy as np


def compute_freestream_direction(alpha_deg: float, beta_deg: float) -> np.ndarray:
    """
    Compute the freestream velocity direction vector from angle of attack and sideslip angle.
    
    Args:
        alpha_deg: Angle of attack in degrees
        beta_deg: Sideslip angle in degrees
        
    Returns:
        Normalized velocity direction vector [x, y, z]
    """
    alpha = np.deg2rad(alpha_deg)
    beta = np.deg2rad(beta_deg)
    vector = np.array(
        [np.cos(alpha) * np.cos(beta), -np.sin(beta), np.sin(alpha) * np.cos(beta)],
        dtype=float,
    )
    vector /= np.linalg.norm(vector)
    return vector


def compute_lift_direction(alpha_deg: float) -> np.ndarray:
    """
    Compute the lift direction vector from angle of attack.
    
    The lift direction is perpendicular to the freestream direction in the x-z plane.
    
    Args:
        alpha_deg: Angle of attack in degrees
        
    Returns:
        Normalized lift direction vector [x, y, z]
    """
    alpha = np.deg2rad(alpha_deg)
    vector = np.array([-np.sin(alpha), 0.0, np.cos(alpha)], dtype=float)
    vector /= np.linalg.norm(vector)
    return vector

