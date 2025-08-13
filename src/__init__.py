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
from .plot_utils import plot_ring_mesh, plot_matrices, plot_mesh_statistics, save_mesh_plots
from .logging_config import setup_logging, get_logger, log_performance, log_performance_conditional
from .pyslsqp_optimizer import PySLSQPOptimizer, RefinementTriggered
from .projection_iterative import (
    orthogonal_projection_iterative,
    orthogonal_projection_direct,
    validate_projection_result,
    create_initial_condition_with_projection
)

__all__ = [
    "Config",
    "RingMesh", 
    "plot_ring_mesh",
    "plot_matrices", 
    "plot_mesh_statistics",
    "save_mesh_plots",
    "setup_logging",
    "get_logger",
    "log_performance",
    "log_performance_conditional",
    "PySLSQPOptimizer",
    "RefinementTriggered",
    "orthogonal_projection_iterative",
    "orthogonal_projection_direct",
    "validate_projection_result",
    "create_initial_condition_with_projection"
] 