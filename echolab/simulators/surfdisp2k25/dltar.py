"""
Compute the period equation for Love and Rayleigh waves.

This module contains the dltar, dltar1, and dltar4 functions, which are pure Python
implementations of the dltar_, dltar1_, and dltar4_ Fortran functions.
"""

# Imports
import numpy as np
from typing import List, Union, Tuple
from .var import var
from .dnka import dnka
from .normc import normc


def dltar1(
        wvno: float,
        omega: float,
        d: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        rho: np.ndarray,
        rtp: np.ndarray,
        dtp: np.ndarray,
        btp: np.ndarray,
        mmax: int,
        llw: int,
        twopi: int
) -> float:
    """
    Compute the period equation for Love waves.
    
    This is a pure Python implementation of the dltar1_ Fortran function.
    
    Parameters
    ----------
    wvno : float
        Wave number in rad/km.
    omega : float
        Angular frequency in rad/s.
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
        Value of the period equation for Love waves.
    """
    # print(">>>>>>>>>>>>>>>>>>>>>>")
    # print("Calling dltar1 ...")
    # print(f"wvno={wvno}")
    # print(f"omega={omega}")
    # print(f"d={d}")
    # print(f"a={a}")
    # print(f"b={b}")
    # print(f"rho={rho}")
    # print(f"rtp={rtp}")
    # print(f"dtp={dtp}")
    # print(f"btp={btp}")
    # print(f"mmax={mmax}")
    # print(f"llw={llw}")
    # print(f"twopi={twopi}")
    # Determine if there's a water layer
    llw = 1
    if b[0] <= 0.0:
        llw = 2
    # end if
    
    # Haskell-Thompson love wave formulation from halfspace to surface
    # Using the exact same variable names and calculations as the Fortran code
    beta1 = float(b[mmax-1])
    rho1 = float(rho[mmax-1])
    xkb = omega / beta1
    wvnop = wvno + xkb
    wvnom = abs(wvno - xkb)
    rb = np.sqrt(wvnop * wvnom)
    e1 = rho1 * rb
    e2 = 1.0 / (beta1 * beta1)
    mmm1 = mmax - 1
    
    # Loop from bottom layer to top (or water layer)
    for m in range(mmm1, llw-1, -1):
        beta1 = float(b[m])
        rho1 = float(rho[m])
        xmu = rho1 * beta1 * beta1
        xkb = omega / beta1
        wvnop = wvno + xkb
        wvnom = abs(wvno - xkb)
        rb = np.sqrt(wvnop * wvnom)
        q = float(d[m]) * rb
        
        # Calculate sinq, y, z, cosq based on the relationship between wvno and xkb
        if wvno < xkb:
            # Propagating case
            sinq = np.sin(q)
            y = sinq / rb
            z = -rb * sinq
            cosq = np.cos(q)
        elif wvno == xkb:
            # Special case: wvno equals xkb
            cosq = 1.0
            y = float(d[m])
            z = 0.0
        else:
            # Evanescent case
            fac = 0.0
            if q < 16:
                fac = np.exp(-2.0 * q)
            cosq = (1.0 + fac) * 0.5
            sinq = (1.0 - fac) * 0.5
            y = sinq / rb
            z = rb * sinq
        # end if wvno
        
        # Calculate the new values of e1 and e2
        e10 = e1 * cosq + e2 * xmu * z
        e20 = e1 * y / xmu + e2 * cosq
        
        # Normalize to prevent overflow
        xnor = abs(e10)
        ynor = abs(e20)
        
        if ynor > xnor:
            xnor = ynor
        # end if

        if xnor < 1.0e-40:
            xnor = 1.0
        # end if
        
        e1 = e10 / xnor
        e2 = e20 / xnor
    # end for m
    
    # Return the final value of e1
    # This is the value of the period equation for Love waves
    # The sign is determined by the calculations above and must match the expected values
    # print(f"e1={e1}")
    # print(">>>>>>>>>>>>>>>>>>>>>>")
    return e1
# end def dltar1


def dltar4(
        wvno: float,
        omega: float,
        d: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        rho: np.ndarray,
        rtp: np.ndarray,
        dtp: np.ndarray,
        btp: np.ndarray,
        mmax: int,
        llw: int,
        twopi: int
) -> float:
    """
    Compute the period equation for Rayleigh waves.
    
    This is a pure Python implementation of the dltar4_ Fortran function.
    
    Parameters
    ----------
    wvno : float
        Wave number in rad/km.
    omega : float
        Angular frequency in rad/s.
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
        Value of the period equation for Rayleigh waves.
    """
    # print("$$$$$$$$$$$$$$$$$$$$$$$$$$")
    # print("Entering dltar4")
    # Make sure omega is not too small
    if omega < 1.0e-4:
        omega = 1.0e-4
    # end if
    
    # Calculate wave number squared
    wvno2 = wvno * wvno
    
    # Calculate parameters for the bottom half-space
    xka = omega / a[mmax-1]
    xkb = omega / b[mmax-1]
    wvnop = wvno + xka
    wvnom = abs(wvno - xka)
    ra = np.sqrt(wvnop * wvnom)
    wvnop = wvno + xkb
    wvnom = abs(wvno - xkb)
    rb = np.sqrt(wvnop * wvnom)
    t = b[mmax-1] / omega
    
    # E matrix for the bottom half-space
    gammk = 2.0 * t * t
    gam = gammk * wvno2
    gamm1 = gam - 1.0
    rho1 = rho[mmax-1]
    
    e = np.zeros(5)
    e[0] = rho1 * rho1 * (gamm1 * gamm1 - gam * gammk * ra * rb)
    e[1] = -rho1 * ra
    e[2] = rho1 * (gamm1 - gammk * ra * rb)
    e[3] = rho1 * rb
    e[4] = wvno2 - ra * rb
    # print(f"wvno2 = {wvno2}")
    # print(f"xka = {xka}")
    # print(f"xkb = {xkb}")
    # print(f"wvnop = {wvnop}")
    # print(f"wvnom = {wvnom}")
    # print(f"ra = {ra}")
    # print(f"rb = {rb}")
    # print(f"t = {t}")
    # print(f"gammk = {gammk}")
    # print(f"gam = {gam}")
    # print(f"gamm1 = {gamm1}")
    # print(f"rho1 = {rho1}")
    # print(f"e = {e}")
    # print("--")
    # Matrix multiplication from bottom layer upward
    mmm1 = mmax - 2
    # print(f"mmm1-1 = {mmm1}")
    # print(f"llw = {llw-1}")
    for m in range(mmm1, llw-2, -1):
        xka = omega / a[m]
        xkb = omega / b[m]
        t = b[m] / omega
        gammk = 2.0 * t * t
        gam = gammk * wvno2
        wvnop = wvno + xka
        wvnom = abs(wvno - xka)
        ra = np.sqrt(wvnop * wvnom)
        wvnop = wvno + xkb
        wvnom = abs(wvno - xkb)
        rb = np.sqrt(wvnop * wvnom)
        dpth = d[m]
        rho1 = rho[m]
        p = ra * dpth
        q = rb * dpth
        beta = b[m]
        # print(f"xka = {xka}")
        # print(f"xkb = {xkb}")
        # print(f"t = {t}")
        # print(f"gammk = {gammk}")
        # print(f"gam = {gam}")
        # print(f"wvnop = {wvnop}")
        # print(f"wvnom = {wvnom}")
        # print(f"ra = {ra}")
        # print(f"rb = {rb}")
        # print(f"dpth = {dpth}")
        # print(f"rho1 = {rho1}")
        # print(f"p = {p}")
        # print(f"q = {q}")
        # Evaluate variables for the compound matrix
        w, cosp, exa, a0, cpcq, cpy, cpz, cqw, cqx, xy, xz, wy, wz = var(
            p=p,
            q=q,
            ra=ra,
            rb=rb,
            wvno=wvno,
            xka=xka,
            xkb=xkb,
            dpth=dpth
        )
        
        # Evaluate Dunkin's matrix
        ca = dnka(
            wvno2=wvno2,
            gam=gam,
            gammk=gammk,
            rho=rho1,
            a0=a0,
            cpcq=cpcq,
            cpy=cpy,
            cpz=cpz,
            cqw=cqw,
            cqx=cqx,
            xy=xy,
            xz=xz,
            wy=wy,
            wz=wz
        )
        
        # Matrix multiplication
        ee = np.zeros(5)
        for i in range(5):
            cr = 0.0
            for j in range(5):
                cr += e[j] * ca[j, i]
            # end for
            ee[i] = cr
        # end for
        
        # Normalize to prevent overflow
        ee, _ = normc(
            ee=ee
        )
        
        # Update e matrix
        e = ee.copy()
    # end for m
    
    # Include water layer if present
    if llw != 1:
        # Water layer at the top
        xka = omega / a[0]
        wvnop = wvno + xka
        wvnom = abs(wvno - xka)
        ra = np.sqrt(wvnop * wvnom)
        dpth = d[0]
        rho1 = rho[0]
        p = ra * dpth
        beta = b[0]
        
        # Calculate variables for water layer
        znul = 1.0e-5
        w, cosp, exa, a0, cpcq, cpy, cpz, cqw, cqx, xy, xz, wy, wz = var(
            p=p,
            q=znul,
            ra=ra,
            rb=znul,
            wvno=wvno,
            xka=xka,
            xkb=znul,
            dpth=dpth
        )
        
        w0 = -rho1 * w
        # print(f"out = {cosp * e[0] + w0 * e[1]}")
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$")
        return cosp * e[0] + w0 * e[1]
    else:
        # print(f"out = {e[0]}")
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$")
        return e[0]
    # end if llw

# end def dltar4


def dltar(
        wvno: float,
        omega: float,
        kk: int,
        d: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        rho: np.ndarray,
        rtp: np.ndarray,
        dtp: np.ndarray,
        btp: np.ndarray,
        mmax: int,
        llw: int,
        twopi: int
) -> float:
    """
    Compute the period equation for Love or Rayleigh waves.
    
    This is a pure Python implementation of the dltar_ Fortran function.
    
    Parameters
    ----------
    wvno : float
        Wave number in rad/km.
    omega : float
        Angular frequency in rad/s.
    kk : int
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
        Value of the period equation for the specified wave type.
    """
    # print(">>>>>>>>>>>>>>>>>>>>>>")
    # print("Calling dltar ...")
    # print(f"wvno={wvno}")
    # print(f"omega={omega}")
    # print(f"d={d}")
    # print(f"a={a}")
    # print(f"b={b}")
    # print(f"rho={rho}")
    # print(f"rtp={rtp}")
    # print(f"dtp={dtp}")
    # print(f"btp={btp}")
    # print(f"mmax={mmax}")
    # print(f"llw={llw}")
    # print(f"twopi={twopi}")
    # Check if kk is valid
    if kk not in [1, 2]:
        raise ValueError("kk must be 1 (Love waves) or 2 (Rayleigh waves)")
    # end if
    
    # Dispatch to the appropriate function
    if kk == 1:
        # print("Love wave period equation")
        # Love wave period equation
        return dltar1(
            wvno=wvno,
            omega=omega,
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
    else:  # kk == 2
        # print("Rayleigh wave period equation")
        # Rayleigh wave period equation
        return dltar4(
            wvno=wvno,
            omega=omega,
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
    # end if kk
# end def dltar
