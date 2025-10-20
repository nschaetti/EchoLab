"""
Normalize vectors to control over/underflow.

This module contains the normc function, which is a pure Python implementation
of the normc_ Fortran subroutine.
"""

import numpy as np
from typing import List, Union, Tuple


def normc(
        ee: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Normalize vectors to control over/underflow.
    
    This is a pure Python implementation of the normc_ Fortran subroutine.
    
    Parameters
    ----------
    ee : array_like
        Input array of length 5 to be normalized.
    
    Returns
    -------
    tuple
        A tuple containing:
        - ee_norm: array_like, the normalized array
        - ex: float, the natural logarithm of the normalization factor
    """
    # Make sure ee is a numpy array of the right type and shape
    ee = np.asarray(ee, dtype=np.float64)
    if ee.shape != (5,):
        raise ValueError(f"Expected ee to be an array of shape (5,), got {ee.shape}")
    # end if
    
    # Create a copy of the input array to avoid modifying the original
    ee_copy = ee.copy()
    
    # Initialize the normalization factor
    ex = 0.0
    
    # Find the maximum absolute value in the vector
    t1 = 0.0
    for i in range(5):
        if abs(ee_copy[i]) > t1:
            t1 = abs(ee_copy[i])
        # end if
    # end for
    
    # If the maximum is very small, set it to 1.0 to avoid division by zero
    if t1 < 1.0e-40:
        t1 = 1.0
    # end if
    
    # Normalize the vector by the maximum value
    for i in range(5):
        t2 = ee_copy[i]
        t2 = t2 / t1
        ee_copy[i] = t2
    # end for
    
    # Store the normalization factor in exponential form
    ex = np.log(t1)
    
    # Return the normalized array and the normalization factor
    return ee_copy, ex
# end normc

