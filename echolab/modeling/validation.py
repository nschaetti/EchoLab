"""
Validation utilities for velocity models.

This module provides functions for validating velocity models based on various
criteria such as value ranges, entropy, and uniqueness of values.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def entropy_score(model: np.ndarray, n_bins: int = 64) -> float:
    """
    Calculate the Shannon entropy of the model (discretized histogram).
    """
    hist, _ = np.histogram(model, bins=n_bins, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))
# end entropy_score


def is_valid_model(
    model: np.ndarray,
    min_v: float = 1000,
    max_v: float = 5000,
    zero_thresh: float = 0.01,
    unique_thresh: float = 0.99,
    entropy_thresh: float = 1.0,
    verbose: bool = False,
) -> bool:
    """
    Check if a velocity model is geologically valid.
    """
    if not np.all((model >= min_v) & (model <= max_v)):
        if verbose:
            print(f"[!] Values outside range [{min_v}, {max_v}]")
        return False
    # end if

    if np.isnan(model).any() or np.isinf(model).any():
        if verbose:
            print("[!] NaN or Inf detected")
        return False
    # end if

    zero_ratio = np.mean(model < 1e-3)
    if zero_ratio > zero_thresh:
        if verbose:
            print(f"[!] {zero_ratio * 100:.2f}% of values ≈ 0")
        return False
    # end if

    values, counts = np.unique(model, return_counts=True)
    ratios = counts / model.size
    max_ratio = np.max(ratios)
    if max_ratio > unique_thresh:
        if verbose:
            dominant_val = values[np.argmax(ratios)]
            print(
                f"[!] {max_ratio * 100:.2f}% of the model is occupied by value {dominant_val:.1f}"
            )
        return False
    # end if

    entropy_value = entropy_score(model)
    if entropy_value < entropy_thresh:
        if verbose:
            print(f"[!] Entropy too low: H = {entropy_value:.2f}")
        return False
    # end if

    return True
# end is_valid_model

