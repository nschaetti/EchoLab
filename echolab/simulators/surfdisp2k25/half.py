"""
Interval halving method for refining root.

This module contains the half function, which is a pure Python implementation
of the half_ Fortran subroutine.
"""

import numpy as np
from typing import List, Union, Tuple
from .dltar import dltar

def half(
        c1: float,
        c2: float,
        omega: float,
        ifunc: int,
        d: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        rho: np.ndarray,
        rtp: np.ndarray,
        dtp: np.ndarray,
        btp: np.ndarray,
        mmax: int,
        llw: int,
        twopi: float
) -> Tuple[float, float]:
    """
    Interval halving method for refining root.
    
    This is a pure Python implementation of the half_ Fortran subroutine.
    
    Parameters
    ----------
    c1 : float
        Lower bound of the interval.
    c2 : float
        Upper bound of the interval.
    omega : float
        Angular frequency in rad/s.
    ifunc : int
        Wave type: 1 for Love waves, 2 for Rayleigh waves.
    d : array_like
        Layer thicknesses in km.
    a : array_like
        P-wave velocities in km/s.
    b : array_like
        S-wave velocities in km/s.
    rho : array_like
        Densities in g/cm^3.
    
    Returns
    -------
    tuple
        A tuple containing:
        - c3: float, the midpoint of the interval (c1 + c2) / 2.
        - del3: float, the value of the period equation at c3.
    """
    # Check if ifunc is valid
    if ifunc not in [1, 2]:
        raise ValueError("ifunc must be 1 (Love waves) or 2 (Rayleigh waves)")
    # end if

    # Interval halving method
    c3 = 0.5 * (c1 + c2)
    wvno = omega / c3
    del3 = dltar(
        wvno=wvno,
        omega=omega,
        kk=ifunc,
        d=d,
        a=a,
        b=b,
        rho=rho,
        rtp=rtp,
        dtp=dtp,
        btp=btp,
        mmax=mmax,
        llw=llw,
        twopi=twopi,
    )
    
    return c3, del3
# end def half
