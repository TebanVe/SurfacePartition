"""
Perimeter optimization with area constraints for partition refinement.

This module implements the constrained optimization algorithm from Section 5 of the paper.
It minimizes total perimeter while maintaining equal area constraints using scipy's SLSQP.

TERMINOLOGY (following paper Section 5):
- "cell": Partition region (what we optimize for equal areas)
- "triangle": Mesh triangle element (computational discretization)
- "edge": Mesh triangle edge (computational discretization)

Main features:
- Objective: Minimize total perimeter (including Steiner tree contributions)
- Constraints: Equal area for all partition cells (within tolerance)
- Bounds: λ ∈ [0, 1] for all variable points
- Gradients: Analytical for perimeter and area
- Topology switching: Detect and handle boundary cases
"""

import numpy as np
from scipy.optimize import minimize, OptimizeResult
from typing import Dict, Optional, Tuple
import time

try:
    from ..logging_config import get_logger
    from .tri_mesh import TriMesh
    from .contour_partition import PartitionContour
    from .area_calculator import AreaCalculator
    from .perimeter_calculator import PerimeterCalculator
    from .steiner_handler import SteinerHandler
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from logging_config import get_logger
    from core.tri_mesh import TriMesh
    from core.contour_partition import PartitionContour
    from core.area_calculator import AreaCalculator
    from core.perimeter_calculator import PerimeterCalculator
    from core.steiner_handler import SteinerHandler


class PerimeterOptimizer:
    """
    Constrained perimeter minimization optimizer for partition refinement.
    
    Takes zigzagged contours from indicator functions and optimizes variable
    point positions to minimize total perimeter while preserving equal areas.
    
    Attributes:
        mesh: The underlying TriMesh
        partition: PartitionContour with variable points
        target_area: Target area for each partition cell
        area_calc: AreaCalculator for computing areas and gradients
        perim_calc: PerimeterCalculator for computing perimeters and gradients
        steiner_handler: SteinerHandler for triple point management
        logger: Logger instance
    """
    
    def __init__(self, partition: PartitionContour, mesh: TriMesh, target_area: float):
        """
        Initialize perimeter optimizer.
        
        Args:
            partition: PartitionContour with extracted contours
            mesh: TriMesh object
            target_area: Target area for each partition cell (total_area / n_cells)
        """
        self.mesh = mesh
        self.partition = partition
        self.target_area = float(target_area)
        self.logger = get_logger(__name__)
        
        # Initialize calculators
        self.area_calc = AreaCalculator(mesh, partition)
        self.perim_calc = PerimeterCalculator(mesh, partition)
        self.steiner_handler = SteinerHandler(mesh, partition)
        
        # Optimization state
        self.iteration = 0
        self.objective_history = []
        self.constraint_violation_history = []
        
        self.logger.info(f"Initialized PerimeterOptimizer:")
        self.logger.info(f"  {partition.n_cells} partition cells")
        self.logger.info(f"  {len(partition.variable_points)} variable points")
        self.logger.info(f"  {len(self.steiner_handler.triple_points)} triple points")
        self.logger.info(f"  Target area per partition cell: {target_area:.6f}")
    
    def objective(self, lambda_vec: np.ndarray) -> float:
        """
        Compute total perimeter (objective function to minimize).
        
        Includes both regular segment perimeters and Steiner tree contributions.
        
        Args:
            lambda_vec: Current variable point parameters
            
        Returns:
            Total perimeter length
        """
        self.partition.set_variable_vector(lambda_vec)
        
        # Regular perimeter from contours
        regular_perimeter = self.perim_calc.compute_total_perimeter(lambda_vec)
        
        # Steiner tree contributions from triple points
        steiner_perimeter = self.steiner_handler.get_total_perimeter_contribution()
        
        total = regular_perimeter + steiner_perimeter
        
        return total
    
    def objective_gradient(self, lambda_vec: np.ndarray) -> np.ndarray:
        """
        Compute gradient of objective function ∂(perimeter)/∂λ.
        
        Args:
            lambda_vec: Current variable point parameters
            
        Returns:
            Gradient array of shape (n_variable_points,)
        """
        self.partition.set_variable_vector(lambda_vec)
        
        # Regular perimeter gradient (analytical)
        regular_gradient = self.perim_calc.compute_total_perimeter_gradient(lambda_vec)
        
        # Steiner tree gradient (finite differences, as suggested in paper)
        steiner_gradient = self.steiner_handler.compute_total_gradient_finite_difference()
        
        total_gradient = regular_gradient + steiner_gradient
        
        return total_gradient
    
    def constraint_area_equality(self, lambda_vec: np.ndarray) -> np.ndarray:
        """
        Compute area constraint violations: Area_i - target_area.
        
        For n cells, we constrain n-1 cells (last one is determined by conservation).
        
        Args:
            lambda_vec: Current variable point parameters
            
        Returns:
            Constraint violations array of shape (n_cells - 1,)
        """
        self.partition.set_variable_vector(lambda_vec)
        
        # Compute regular areas
        areas = self.area_calc.compute_all_cell_areas(lambda_vec)
        
        # Add Steiner tree area contributions
        steiner_areas = self.steiner_handler.get_total_area_contribution()
        for cell_idx, area_contrib in steiner_areas.items():
            if cell_idx < len(areas):
                areas[cell_idx] += area_contrib
        
        # Return constraint violations for first n-1 cells
        return areas[:-1] - self.target_area
    
    def constraint_area_jacobian(self, lambda_vec: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of area constraints ∂(Area_i - target)/∂λ.
        
        Per paper section 5: Uses finite differences for Steiner area gradients.
        
        Args:
            lambda_vec: Current variable point parameters
            
        Returns:
            Jacobian array of shape (n_cells - 1, n_variable_points)
        """
        self.partition.set_variable_vector(lambda_vec)
        
        # Regular area Jacobian (from boundary triangles, analytical)
        jacobian = self.area_calc.compute_area_jacobian(lambda_vec)
        
        # Add Steiner tree area gradients (finite differences, per paper line 366)
        steiner_gradients = self.steiner_handler.compute_area_gradients_finite_difference(
            self.mesh, eps=1e-7
        )
        
        # Add Steiner contributions to jacobian (only first n_cells-1 rows)
        for cell_idx in range(self.partition.n_cells - 1):
            jacobian[cell_idx, :] += steiner_gradients[cell_idx]
        
        return jacobian
    
    def _callback(self, *args, **kwargs):
        """
        Callback function called at each optimization iteration.
        
        Handles different signatures from different optimization methods:
        - SLSQP: callback(xk) - xk is the parameter vector
        - trust-constr: callback(state) or callback(xk, res) - need to extract xk
        
        Args:
            *args: Variable arguments - first arg is typically xk or state object
            **kwargs: Keyword arguments (not used)
        """
        # Extract xk from different callback signatures
        if len(args) == 0:
            return  # No arguments, skip
        
        # For SLSQP: args[0] is xk (numpy array)
        # For trust-constr: scipy wraps and calls with (xk, res) where xk is np.ndarray
        # For trust-constr direct: args[0] might be state object with .x attribute
        if len(args) >= 2:
            # Multiple arguments: trust-constr wrapped signature (xk, res)
            # First arg should be xk (numpy array)
            if isinstance(args[0], np.ndarray):
                xk = args[0]
            else:
                # Try second argument
                xk = args[1].x if hasattr(args[1], 'x') else args[0]
        elif isinstance(args[0], np.ndarray):
            # Single numpy array: SLSQP signature
            xk = args[0]
        elif hasattr(args[0], 'x'):
            # trust-constr state object with .x attribute
            xk = args[0].x
        else:
            # Fallback: try to get x attribute
            xk = getattr(args[0], 'x', None)
            if xk is None:
                return  # Can't extract xk, skip
        
        self.iteration += 1
        
        # Compute and log progress
        obj = self.objective(xk)
        constraints = self.constraint_area_equality(xk)
        max_violation = float(np.max(np.abs(constraints)))
        
        self.objective_history.append(obj)
        self.constraint_violation_history.append(max_violation)
        
        if self.iteration % 10 == 0:
            self.logger.info(f"Iteration {self.iteration}: "
                           f"Perimeter={obj:.6f}, MaxViolation={max_violation:.2e}")
    
    def optimize(self, max_iter: int = 1000, tol: float = 1e-7,
                method: str = 'SLSQP') -> OptimizeResult:
        """
        Run constrained perimeter optimization.
        
        Uses scipy.optimize.minimize with:
        - Method: SLSQP (Sequential Least Squares Programming) by default
        - Objective: Total perimeter
        - Constraints: Equal area for each cell (n-1 constraints)
        - Bounds: λ ∈ [0, 1] for all variable points
        
        Args:
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            method: Optimization method ('SLSQP' or 'trust-constr')
            
        Returns:
            scipy OptimizeResult object
        """
        self.logger.info(f"Starting perimeter optimization with method={method}")
        self.logger.info(f"  max_iter={max_iter}, tol={tol}")
        
        # Initial guess: current λ values (all 0.5 from initialization)
        lambda0 = self.partition.get_variable_vector()
        
        # Box constraints: λ ∈ [0, 1]
        bounds = [(0.0, 1.0) for _ in lambda0]
        
        # Area equality constraints
        constraints = {
            'type': 'eq',
            'fun': self.constraint_area_equality,
            'jac': self.constraint_area_jacobian
        }
        
        # Initial objective and constraint values
        obj0 = self.objective(lambda0)
        constr0 = self.constraint_area_equality(lambda0)
        max_viol0 = float(np.max(np.abs(constr0)))
        
        self.logger.info(f"Initial state:")
        self.logger.info(f"  Perimeter: {obj0:.6f}")
        self.logger.info(f"  Max constraint violation: {max_viol0:.2e}")
        
        # Run optimization
        start_time = time.time()
        
        # Set method-specific options
        if method == 'SLSQP':
            options = {'maxiter': max_iter, 'ftol': tol, 'disp': True}
        elif method == 'trust-constr':
            # trust-constr uses 'gtol' for gradient tolerance and 'xtol' for x tolerance
            # 'maxiter' is also supported
            options = {'maxiter': max_iter, 'gtol': tol, 'xtol': tol, 'disp': True}
        else:
            # Default options for other methods
            options = {'maxiter': max_iter, 'disp': True}
        
        result = minimize(
            fun=self.objective,
            x0=lambda0,
            method=method,
            jac=self.objective_gradient,
            bounds=bounds,
            constraints=constraints,
            callback=self._callback,
            options=options
        )
        
        elapsed_time = time.time() - start_time
        
        # Update partition with optimized parameters
        self.partition.set_variable_vector(result.x)
        
        # Final statistics
        final_obj = self.objective(result.x)
        final_constr = self.constraint_area_equality(result.x)
        final_max_viol = float(np.max(np.abs(final_constr)))
        
        self.logger.info(f"Optimization completed in {elapsed_time:.2f}s")
        self.logger.info(f"  Success: {result.success}")
        self.logger.info(f"  Message: {result.message}")
        self.logger.info(f"  Iterations: {result.nit}")
        self.logger.info(f"  Function evaluations: {result.nfev}")
        self.logger.info(f"  Final perimeter: {final_obj:.6f}")
        self.logger.info(f"  Perimeter reduction: {obj0 - final_obj:.6f} ({100*(obj0-final_obj)/obj0:.2f}%)")
        self.logger.info(f"  Final max constraint violation: {final_max_viol:.2e}")
        
        return result
    
    def get_optimization_info(self, result: OptimizeResult) -> Dict:
        """
        Extract optimization metadata for saving.
        
        Args:
            result: scipy OptimizeResult object
            
        Returns:
            Dictionary with optimization information
        """
        final_perimeter = self.objective(result.x)
        final_areas = self.area_calc.compute_all_cell_areas(result.x)
        
        # Add Steiner contributions to areas
        steiner_areas = self.steiner_handler.get_total_area_contribution()
        for cell_idx, area_contrib in steiner_areas.items():
            if cell_idx < len(final_areas):
                final_areas[cell_idx] += area_contrib
        
        return {
            'success': bool(result.success),
            'n_iterations': int(result.nit),
            'n_function_evals': int(result.nfev),
            'final_perimeter': float(final_perimeter),
            'final_areas': final_areas.tolist(),
            'target_area': float(self.target_area),
            'max_area_violation': float(np.max(np.abs(final_areas - self.target_area))),
            'message': str(result.message),
            'objective_history': self.objective_history,
            'constraint_violation_history': self.constraint_violation_history
        }
    
    def check_topology_switches_needed(self, tol: float = 1e-3) -> Tuple[bool, Dict]:
        """
        Check if topology switches are needed after optimization.
        
        Returns whether any λ parameters are near boundaries (0 or 1),
        indicating that variable points want to move to adjacent edges.
        
        Args:
            tol: Threshold for considering a point at boundary
            
        Returns:
            (switches_needed, switch_info) where switch_info contains details
        """
        boundary_points = self.partition.get_boundary_variable_points(tol)
        boundary_triple_points = self.steiner_handler.get_boundary_triple_points(tol)
        
        switches_needed = len(boundary_points) > 0 or len(boundary_triple_points) > 0
        
        switch_info = {
            'n_boundary_points': len(boundary_points),
            'boundary_point_indices': boundary_points,
            'n_boundary_triple_points': len(boundary_triple_points),
            'boundary_triple_point_indices': [tp.triangle_idx for tp in boundary_triple_points]
        }
        
        if switches_needed:
            self.logger.info(f"Topology switches needed:")
            self.logger.info(f"  {len(boundary_points)} variable points near boundaries")
            self.logger.info(f"  {len(boundary_triple_points)} triple points near boundaries")
        
        return switches_needed, switch_info

