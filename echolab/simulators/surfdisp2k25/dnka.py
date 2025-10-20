"""
Calculate Dunkin's matrix for layered media.

This module contains the dnka function, which is a pure Python implementation
of the dnka_ Fortran subroutine.
"""

# Imports
import numpy as np
from typing import List, Union, Tuple


def dnka(
        wvno2: float,
        gam: float,
        gammk: float,
        rho: float,
        a0: float,
        cpcq: float,
        cpy: float,
        cpz: float,
        cqw: float,
        cqx: float,
        xy: float,
        xz: float,
        wy: float,
        wz: float
) -> np.ndarray:
    """
    Calculate Dunkin's matrix for layered media.
    
    This is a pure Python implementation of the dnka_ Fortran subroutine.
    
    Parameters
    ----------
    wvno2 : float
        Square of horizontal wavenumber in (rad/km)^2.
    gam : float
        Parameter gamma.
    gammk : float
        Parameter gamma_k.
    rho : float
        Density in g/cm^3.
    a0 : float
        Parameter a0 from var function.
    cpcq : float
        Parameter cpcq from var function.
    cpy : float
        Parameter cpy from var function.
    cpz : float
        Parameter cpz from var function.
    cqw : float
        Parameter cqw from var function.
    cqx : float
        Parameter cqx from var function.
    xy : float
        Parameter xy from var function.
    xz : float
        Parameter xz from var function.
    wy : float
        Parameter wy from var function.
    wz : float
        Parameter wz from var function.
    
    Returns
    -------
    numpy.ndarray
        5x5 Dunkin's matrix used in the calculation of dispersion curves for Rayleigh waves.
    """
    # Constants
    one = 1.0
    two = 2.0
    
    # Calculate intermediate values
    gamm1 = gam - one
    twgm1 = gam + gamm1
    gmgmk = gam * gammk
    gmgm1 = gam * gamm1
    gm1sq = gamm1 * gamm1
    rho2 = rho * rho
    a0pq = a0 - cpcq
    
    # Initialize the output matrix
    ca = np.zeros((5, 5))
    
    # Calculate Dunkin's matrix elements
    ca[0, 0] = cpcq - two * gmgm1 * a0pq - gmgmk * xz - wvno2 * gm1sq * wy
    ca[0, 1] = (wvno2 * cpy - cqx) / rho
    ca[0, 2] = -(twgm1 * a0pq + gammk * xz + wvno2 * gamm1 * wy) / rho
    ca[0, 3] = (cpz - wvno2 * cqw) / rho
    ca[0, 4] = -(two * wvno2 * a0pq + xz + wvno2 * wvno2 * wy) / rho2
    
    ca[1, 0] = (gmgmk * cpz - gm1sq * cqw) * rho
    ca[1, 1] = cpcq
    ca[1, 2] = gammk * cpz - gamm1 * cqw
    ca[1, 3] = -wz
    ca[1, 4] = ca[0, 3]
    
    ca[3, 0] = (gm1sq * cpy - gmgmk * cqx) * rho
    ca[3, 1] = -xy
    ca[3, 2] = gamm1 * cpy - gammk * cqx
    ca[3, 3] = ca[1, 1]
    ca[3, 4] = ca[0, 1]
    
    ca[4, 0] = -(two * gmgmk * gm1sq * a0pq + gmgmk * gmgmk * xz + gm1sq * gm1sq * wy) * rho2
    ca[4, 1] = ca[3, 0]
    ca[4, 2] = -(gammk * gamm1 * twgm1 * a0pq + gam * gammk * gammk * xz + gamm1 * gm1sq * wy) * rho
    ca[4, 3] = ca[1, 0]
    ca[4, 4] = ca[0, 0]
    
    t = -two * wvno2
    ca[2, 0] = t * ca[4, 2]
    ca[2, 1] = t * ca[3, 2]
    ca[2, 2] = a0 + two * (cpcq - ca[0, 0])
    ca[2, 3] = t * ca[1, 2]
    ca[2, 4] = t * ca[0, 2]
    
    return ca
# end def dnka
