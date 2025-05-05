"""
Recommendation system module.

This package contains implementations of various recommendation algorithms.
"""

from .als_spark import SparkALS
from .numpy_als import NumPyALS

__all__ = ["SparkALS", "NumPyALS"]
