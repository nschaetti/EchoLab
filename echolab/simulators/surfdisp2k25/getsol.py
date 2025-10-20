"""
Python implementation of the getsol subroutine from surfdisp96.

This module provides a Python implementation of the getsol subroutine
from the surfdisp96 Fortran code, which is responsible for bracketing
dispersion curves and then refining them.
"""
import math

import numpy as np
from typing import Tuple, List, Union
import compearth.extensions.surfdisp2k25 as sd2k25


_del1st = 0.0


def getsol(
        t1: float,
        c1: float,
        clow: float,
        dc: float,
        cm: float,
        betmx: float,
        ifunc: int,
        ifirst: int,
        d: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        rho: np.ndarray,
        rtp: np.ndarray,
        dtp: np.ndarray,
        btp: np.ndarray,
        mmax: int,
        llw: int
) -> Tuple[float, int]:
    """
    Bracket dispersion curve and then refine it.
    
    This is a pure Python implementation of the getsol_ Fortran subroutine.
    
    Parameters
    ----------
    t1 : float
        Period in seconds.
    c1 : float
        Initial phase velocity estimate in km/s.
    clow : float
        Lower bound for phase velocity in km/s.
    dc : float
        Phase velocity increment for search in km/s.
    cm : float
        Minimum phase velocity to consider in km/s.
    betmx : float
        Maximum phase velocity to consider in km/s.
    ifunc : int
        Wave type: 1 for Love waves, 2 for Rayleigh waves.
    ifirst : int
        First call flag: 1 for first call, 0 otherwise.
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
        - c1: float, the refined phase velocity in km/s.
        - iret: int, return code (1 for success, -1 for failure).
    """
    global _del1st

    # Check if ifunc is valid
    if ifunc not in [1, 2]:
        raise ValueError("ifunc must be 1 (Love waves) or 2 (Rayleigh waves)")
    # end if
    # Initialize twopi
    twopi = 2.0 * np.pi

    # Bracket solution
    omega = twopi / t1
    wvno = omega / c1
    
    # Bracket solution
    del1 = sd2k25.dltar(
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
        twopi=twopi
    )

    if ifirst == 1:
        _del1st = del1
    # end if

    plmn = math.copysign(1.0, _del1st) * np.sign(del1)
    
    if ifirst == 1:
        idir = 1
    elif ifirst != 1 and plmn >= 0.0:
        idir = 1
    elif ifirst != 1 and plmn < 0.0:
        idir = -1
    else:
        raise ValueError("ifirst must be 1 or 0.")
    # end if ifirst
    
    # idir indicates the direction of the search for the true phase velocity from the initial estimate.
    # Usually phase velocity increases with period and we always underestimate, so phase velocity should increase
    # (idir = +1). For reversed dispersion, we should look downward from the present estimate. 
    # However, we never go below the floor of clow, when the direction is reversed
    while True:
        if idir > 0:
            c2 = c1 + dc
        else:
            c2 = c1 - dc
        # end if
        
        if c2 <= clow:
            idir = 1
            c1 = clow
            continue
        # end if
        
        omega = twopi / t1
        wvno = omega / c2
        del2 = sd2k25.dltar(
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
            twopi=twopi
        )
        # Changed sign
        if math.copysign(1.0, del1) != math.copysign(1.0, del2):
            # Root bracketed, refine it
            cn = sd2k25.nevill(
                t=t1,
                c1=c1,
                c2=c2,
                del1=del1,
                del2=del2,
                ifunc=ifunc,
                d=d,
                a=a,
                b=b,
                rho=rho,
                rtp=rtp,
                dtp=dtp,
                btp=btp,
                mmax=mmax,
                llw=llw,
                twopi=twopi
            )
            c1 = cn

            # Clamp the refined phase velocity to betmx when it exceeds betmx
            if c1 > betmx:
                return c1, -1
            # end if
            return c1, 1
        # end if np.sign
        
        c1 = c2
        del1 = del2
        
        # Check that c1 is in a region of solutions
        if c1 < cm or c1 >= (betmx + dc):
            return c1, -1
        # end if c1
    # end while
# end def getsol

