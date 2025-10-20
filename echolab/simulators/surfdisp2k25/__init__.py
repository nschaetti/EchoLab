"""
Python implementation of the surfdisp96 extension.

This module provides functions for calculating dispersion curves for surface waves
in layered media. It is a pure Python implementation of the surfdisp96 Fortran code.
"""

import numpy as np

# Import functions from their respective modules
from .getsol import getsol
from .getsolh import getsolh
from .dispsurf2k25 import dispsurf2k25, dispsurf2k25_simulator
from .normc import normc
from .var import var
from .dnka import dnka
from .dltar import dltar, dltar1, dltar4
from .half import half
from .nevill import nevill
from .sphere import sphere

# Define the public API
__all__ = [
    'getsol',
    'getsolh',
    'dispsurf2k25_simulator',
    'dispsurf2k25',
    'normc',
    'var',
    'dnka',
    'dltar',
    'dltar1',
    'dltar4',
    'half',
    'nevill',
    'sphere'
]