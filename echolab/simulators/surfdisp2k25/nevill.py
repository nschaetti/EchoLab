"""
Hybrid method for refining root once it has been bracketed.

This module contains the nevill function, which is a pure Python implementation
of the nevill_ Fortran subroutine.
"""


# Imports
import numpy as np
from typing import List, Union, Tuple
from .half import half
from .dltar import dltar


def nevill(
        t: float,
        c1: float,
        c2: float,
        del1: float,
        del2: float,
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
) -> float:
    """
    Hybrid method for refining root once it has been bracketed.
    
    This is a pure Python implementation of the nevill_ Fortran subroutine.
    
    Parameters
    ----------
    t : float
        Period in seconds.
    c1 : float
        Lower bound of the interval.
    c2 : float
        Upper bound of the interval.
    del1 : float
        Value of the period equation at c1.
    del2 : float
        Value of the period equation at c2.
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
    float
        The refined phase velocity.
    """
    # Check if ifunc is valid
    if ifunc not in [1, 2]:
        raise ValueError("ifunc must be 1 (Love waves) or 2 (Rayleigh waves)")
    # end if
    
    # Calculate angular frequency
    omega = 2.0 * np.pi / t
    
    # Initial guess using interval halving
    c3, del3 = half(
        c1,
        c2,
        omega,
        ifunc,
        d,
        a,
        b,
        rho,
        rtp=rtp,
        dtp=dtp,
        btp=btp,
        mmax=mmax,
        llw=llw,
        twopi=twopi,
    )
    nev = 1
    nctrl = 1
    
    # Arrays for Neville iteration
    x = np.zeros(20)
    y = np.zeros(20)
    
    # Main loop
    while True:
        nctrl += 1
        if nctrl >= 100:
            break
        # end if
        
        # Make sure new estimate is inside the previous values
        # If not, perform interval halving
        if c3 < min(c1, c2) or c3 > max(c1, c2):
            nev = 0
            c3, del3 = half(
                c1=c1,
                c2=c2,
                omega=omega,
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
                twopi=twopi,
            )
        # end if
        
        s13 = del1 - del3
        s32 = del3 - del2
        
        # Define new bounds according to the sign of the period equation
        if np.sign(del3) * np.sign(del1) < 0.0:
            c2 = c3
            del2 = del3
        else:
            c1 = c3
            del1 = del3
        # end if
        
        # Check for convergence using relative error criteria
        if abs(c1 - c2) <= 1e-6 * c1:
            break
        # end if
        
        # If the slopes are not the same between c1, c3 and c3, c2
        # do not use Neville iteration
        if np.sign(s13) != np.sign(s32):
            nev = 0
        # end if
        
        # If the period equation differs by more than a factor of 10
        # use interval halving to avoid poor behavior of polynomial fit
        ss1 = abs(del1)
        s1 = 0.01 * ss1
        ss2 = abs(del2)
        s2 = 0.01 * ss2
        
        if s1 > ss2 or s2 > ss1 or nev == 0:
            c3, del3 = half(
                c1=c1,
                c2=c2,
                omega=omega,
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
                twopi=twopi,
            )
            nev = 1
            m = 1
        else:
            if nev == 2:
                x[m] = c3
                y[m] = del3
            else:
                x[0] = c1
                y[0] = del1
                x[1] = c2
                y[1] = del2
                m = 1
            # end if
            
            # Perform Neville iteration
            try:
                for kk in range(1, m + 1):
                    j = m - kk
                    denom = y[m] - y[j]
                    if abs(denom) < 1.0e-10 * abs(y[m]):
                        raise ValueError("Denominator too small in Neville iteration")
                    # end if
                    x[j] = (-y[j] * x[j+1] + y[m] * x[j]) / denom
                # end for
            except ValueError:
                # If there's an error in Neville iteration, fall back to interval halving
                c3, del3 = half(
                    c1=c1,
                    c2=c2,
                    omega=omega,
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
                    twopi=twopi,
                )
                nev = 1
                m = 1
            else:
                c3 = x[0]
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
                nev = 2
                m += 1
                if m > 10:
                    m = 10
                # end if
            # end try
        # end if s1, s2
    # end while True
    
    # Return the refined phase velocity
    return c3
# end def nevill
