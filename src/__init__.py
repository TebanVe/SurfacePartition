"""
Ring Partition Project

This package implements the optimization method for partitioning a ring (annulus) 
in R² into equal area regions while minimizing the total perimeter.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main classes for easy access
from .config import Config
from .ring_mesh import RingMesh
from .pyslsqp_optimizer import PySLSQPOptimizer
from .find_contours import ContourAnalyzer

__all__ = [
    "Config",
    "RingMesh", 
    "PySLSQPOptimizer",
    "ContourAnalyzer",
] 