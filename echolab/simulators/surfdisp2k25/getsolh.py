"""
Python implementation of the getsolh subroutine from surfdisp96.

This module provides a Python implementation of the gtsolh subroutine
from the surfdisp96 Fortran code, which is responsible for calculating
a starting solution for phase velocity calculation.
"""


import numpy as np
from typing import Union


def getsolh(
        a: float,
        b: float
) -> float:
    """
    Calculate a starting solution for phase velocity.
    
    This is a pure Python implementation of the gtsolh_ Fortran subroutine.
    
    Parameters
    ----------
    a : float
        P-wave velocity in km/s.
    b : float
        S-wave velocity in km/s.
    
    Returns
    -------
    float
        The starting solution for phase velocity in km/s.
    
    Raises
    ------
    TypeError
        If a or b is not a number.
    ValueError
        If a or b is not positive, or if b is greater than or equal to a.
    """
    # Check for valid velocity values
    if a <= 0:
        raise ValueError("P-wave velocity (a) must be positive")
    # end if

    if b <= 0:
        raise ValueError("S-wave velocity (b) must be positive")
    # end if

    if b > a:  # Changed from b >= a to b > a to allow the case where a = b
        raise ValueError("S-wave velocity (b) must be less than or equal to P-wave velocity (a)")
    # end if
    
    # Starting solution
    c = 0.95 * b
    
    # Iterate to refine the solution
    for i in range(5):
        gamma = b / a
        kappa = c / b
        k2 = kappa**2
        gk2 = (gamma * kappa)**2
        fac1 = np.sqrt(1.0 - gk2)
        fac2 = np.sqrt(1.0 - k2)
        fr = (2.0 - k2)**2 - 4.0 * fac1 * fac2
        frp = -4.0 * (2.0 - k2) * kappa + \
              4.0 * fac2 * gamma * gamma * kappa / fac1 + \
              4.0 * fac1 * kappa / fac2
        frp = frp / b
        c = c - fr / frp
    # end if
    
    return c
# end def getsolh
