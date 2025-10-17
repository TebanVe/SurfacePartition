"""
Steiner tree handling for triple points in partition contours.

This module implements the triple point treatment from Section 5 of the paper:
"The empty spaces around triple points". 

When three different regions meet at a triangle, small void spaces are created.
These are filled with Steiner trees that:
1. Connect three variable points on the triangle's edges
2. Meet at an optimal Steiner point that minimizes total edge length
3. Divide the void area among the three adjacent cells

The Steiner point satisfies the Fermat point property: edges meet at 120° angles.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.optimize import minimize

try:
    from ..logging_config import get_logger
    from .tri_mesh import TriMesh
    from .contour_partition import PartitionContour
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from logging_config import get_logger
    from core.tri_mesh import TriMesh
    from core.contour_partition import PartitionContour


class TriplePoint:
    """
    Represents a triple point where three partition regions meet.
    
    At a triple point, three variable points on a triangle's edges form a small
    void space. A Steiner point is computed to optimally connect these three points,
    minimizing the total perimeter contribution.
    
    Attributes:
        triangle_idx: Index of the mesh triangle containing this triple point
        var_point_indices: List of 3 variable point indices on the triangle edges
        cell_indices: List of 3 cell indices that meet here
        steiner_point: Optimal connection point (computed)
        boundary_points: 3D/2D coordinates of the three variable points
    """
    
    def __init__(self, triangle_idx: int, var_point_indices: List[int],
                 partition: PartitionContour):
        """
        Initialize triple point.
        
        Args:
            triangle_idx: Index of triangle in mesh
            var_point_indices: List of 3 variable point indices
            partition: PartitionContour object
        """
        self.triangle_idx = triangle_idx
        self.var_point_indices = var_point_indices
        self.partition = partition
        self.logger = get_logger(__name__)
        
        # Determine which cells meet at this point
        self.cell_indices: List[int] = []
        for vp_idx in var_point_indices:
            vp = partition.variable_points[vp_idx]
            self.cell_indices.extend(list(vp.adjacent_cells))
        # Get unique cells (should be exactly 3)
        self.cell_indices = list(set(self.cell_indices))
        
        if len(self.cell_indices) != 3:
            self.logger.warning(f"Triple point at triangle {triangle_idx} has "
                              f"{len(self.cell_indices)} cells, expected 3")
        
        # Will be computed
        self.steiner_point: Optional[np.ndarray] = None
        self.boundary_points: Optional[List[np.ndarray]] = None
    
    def compute_steiner_point(self) -> np.ndarray:
        """
        Compute optimal Steiner point that minimizes total edge length.
        
        For three points p1, p2, p3, find point S that minimizes:
            f(S) = ||S - p1|| + ||S - p2|| + ||S - p3||
        
        This is the geometric median (Fermat point) problem.
        For a proper Steiner tree, edges meet at 120° angles.
        
        Returns:
            Optimal Steiner point coordinates (2D or 3D)
        """
        # Evaluate the three variable points
        self.boundary_points = [
            self.partition.evaluate_variable_point(vp_idx)
            for vp_idx in self.var_point_indices
        ]
        
        p1, p2, p3 = self.boundary_points
        
        # Initial guess: centroid of the three points
        initial_guess = (p1 + p2 + p3) / 3.0
        
        # Objective: sum of distances
        def objective(S):
            return (np.linalg.norm(S - p1) + 
                   np.linalg.norm(S - p2) + 
                   np.linalg.norm(S - p3))
        
        # Gradient of objective
        def gradient(S):
            grad = np.zeros_like(S)
            for p in [p1, p2, p3]:
                dist = np.linalg.norm(S - p)
                if dist > 1e-12:
                    grad += (S - p) / dist
            return grad
        
        # Optimize
        result = minimize(objective, initial_guess, jac=gradient, method='BFGS',
                         options={'gtol': 1e-8})
        
        self.steiner_point = result.x
        return self.steiner_point
    
    def get_perimeter_contribution(self) -> Dict[int, float]:
        """
        Compute perimeter contribution to each adjacent cell.
        
        Each of the three cells gets two edges from the Steiner tree.
        For example, if cell i is adjacent to variable points vp_j and vp_k,
        it gets edges from vp_j to S and from S to vp_k.
        
        Returns:
            Dict mapping cell_idx -> additional perimeter length
        """
        if self.steiner_point is None:
            self.compute_steiner_point()
        
        contributions = {cell_idx: 0.0 for cell_idx in self.cell_indices}
        
        # For each variable point, add edge to Steiner point for its adjacent cells
        for vp_idx in self.var_point_indices:
            vp_pos = self.partition.evaluate_variable_point(vp_idx)
            edge_length = np.linalg.norm(vp_pos - self.steiner_point)
            
            # This edge contributes to the cells adjacent to this variable point
            vp = self.partition.variable_points[vp_idx]
            for cell_idx in vp.adjacent_cells:
                if cell_idx in contributions:
                    contributions[cell_idx] += edge_length
        
        return contributions
    
    def get_area_contribution(self, mesh: TriMesh) -> Dict[int, float]:
        """
        Compute area contribution to each adjacent cell.
        
        The triangular void is divided into three sub-triangles by the Steiner point.
        Each sub-triangle is assigned to the cell adjacent to its base edge.
        
        Returns:
            Dict mapping cell_idx -> additional area
        """
        if self.steiner_point is None:
            self.compute_steiner_point()
        
        contributions = {cell_idx: 0.0 for cell_idx in self.cell_indices}
        
        # Map each edge (variable point pair) to its adjacent cells
        # Then assign the sub-triangle formed by that edge and Steiner point
        
        # For simplicity, divide area equally among the three cells
        # (More accurate would be to compute actual sub-triangle areas)
        triangle_area = mesh.triangle_areas[self.triangle_idx]
        for cell_idx in self.cell_indices:
            contributions[cell_idx] = triangle_area / 3.0
        
        return contributions
    
    def get_segments(self) -> Dict[int, List[np.ndarray]]:
        """
        Get Steiner tree segments for visualization.
        
        Returns:
            Dict mapping cell_idx -> list of segment arrays (2, D)
        """
        if self.steiner_point is None:
            self.compute_steiner_point()
        
        segments_dict = {cell_idx: [] for cell_idx in self.cell_indices}
        
        # Create segments from each variable point to Steiner point
        for vp_idx in self.var_point_indices:
            vp_pos = self.partition.evaluate_variable_point(vp_idx)
            segment = np.vstack([vp_pos, self.steiner_point])
            
            # Add to cells adjacent to this variable point
            vp = self.partition.variable_points[vp_idx]
            for cell_idx in vp.adjacent_cells:
                if cell_idx in segments_dict:
                    segments_dict[cell_idx].append(segment)
        
        return segments_dict
    
    def compute_gradients_finite_difference(self, eps: float = 1e-6) -> np.ndarray:
        """
        Compute gradients of Steiner tree perimeter w.r.t. λ parameters.
        
        As suggested in the paper, use finite differences for Steiner point gradients
        since analytical derivatives are complex.
        
        Args:
            eps: Perturbation size for finite differences
            
        Returns:
            Gradient array of shape (n_variable_points,)
        """
        n_vars = len(self.partition.variable_points)
        gradient = np.zeros(n_vars)
        
        # Base perimeter contribution
        base_contrib = self.get_perimeter_contribution()
        base_total = sum(base_contrib.values())
        
        # Perturb each λ parameter that affects this triple point
        for vp_idx in self.var_point_indices:
            old_lambda = self.partition.variable_points[vp_idx].lambda_param
            
            # Perturb
            self.partition.variable_points[vp_idx].lambda_param = old_lambda + eps
            
            # Recompute Steiner point and perimeter
            self.steiner_point = None  # Force recomputation
            perturbed_contrib = self.get_perimeter_contribution()
            perturbed_total = sum(perturbed_contrib.values())
            
            # Gradient via finite difference
            gradient[vp_idx] = (perturbed_total - base_total) / eps
            
            # Restore original value
            self.partition.variable_points[vp_idx].lambda_param = old_lambda
        
        # Force recomputation with original values
        self.steiner_point = None
        self.compute_steiner_point()
        
        return gradient
    
    def is_on_triangle_boundary(self, tol: float = 1e-3) -> bool:
        """
        Check if Steiner point is near the triangle boundary.
        
        If true, topology switch may be needed (expand to adjacent triangle).
        
        Args:
            tol: Distance tolerance
            
        Returns:
            True if Steiner point is within tol of any triangle edge
        """
        if self.steiner_point is None:
            self.compute_steiner_point()
        
        # Get triangle vertices
        face = self.partition.mesh.faces[self.triangle_idx]
        v1, v2, v3 = [int(i) for i in face]
        p1 = self.partition.mesh.vertices[v1]
        p2 = self.partition.mesh.vertices[v2]
        p3 = self.partition.mesh.vertices[v3]
        
        # Check distance to each edge
        edges = [(p1, p2), (p2, p3), (p3, p1)]
        for edge_start, edge_end in edges:
            dist = self._point_to_segment_distance(self.steiner_point, edge_start, edge_end)
            if dist < tol:
                return True
        
        return False
    
    def _point_to_segment_distance(self, point: np.ndarray, 
                                   seg_start: np.ndarray, 
                                   seg_end: np.ndarray) -> float:
        """
        Compute distance from point to line segment.
        
        Args:
            point: Query point
            seg_start: Segment start point
            seg_end: Segment end point
            
        Returns:
            Minimum distance
        """
        # Vector from start to end
        v = seg_end - seg_start
        # Vector from start to point
        w = point - seg_start
        
        # Projection parameter
        c1 = np.dot(w, v)
        if c1 <= 0:
            return np.linalg.norm(point - seg_start)
        
        c2 = np.dot(v, v)
        if c1 >= c2:
            return np.linalg.norm(point - seg_end)
        
        # Project onto segment
        b = c1 / c2
        proj = seg_start + b * v
        return np.linalg.norm(point - proj)


class SteinerHandler:
    """
    Manages all triple points in the partition and their Steiner trees.
    
    Attributes:
        mesh: The underlying TriMesh
        partition: PartitionContour with variable points
        triple_points: List of detected TriplePoint objects
    """
    
    def __init__(self, mesh: TriMesh, partition: PartitionContour):
        """
        Initialize Steiner handler and detect triple points.
        
        Args:
            mesh: TriMesh object
            partition: PartitionContour with variable points
        """
        self.mesh = mesh
        self.partition = partition
        self.logger = get_logger(__name__)
        
        # Detect and create triple points
        self.triple_points: List[TriplePoint] = []
        self._detect_triple_points()
    
    def _detect_triple_points(self):
        """Detect all triple points in the partition."""
        triple_point_data = self.partition.identify_triple_points()
        
        for tri_idx, var_point_indices in triple_point_data:
            tp = TriplePoint(tri_idx, var_point_indices, self.partition)
            tp.compute_steiner_point()
            self.triple_points.append(tp)
        
        self.logger.info(f"Detected and initialized {len(self.triple_points)} triple points")
    
    def get_total_perimeter_contribution(self) -> float:
        """
        Compute total perimeter contribution from all Steiner trees.
        
        Returns:
            Sum of perimeter contributions from all triple points
        """
        total = 0.0
        for tp in self.triple_points:
            contrib = tp.get_perimeter_contribution()
            total += sum(contrib.values())
        
        return total
    
    def get_total_area_contribution(self) -> Dict[int, float]:
        """
        Compute total area contribution from all Steiner trees to each cell.
        
        Returns:
            Dict mapping cell_idx -> total additional area
        """
        contributions = {i: 0.0 for i in range(self.partition.n_cells)}
        
        for tp in self.triple_points:
            area_contrib = tp.get_area_contribution(self.mesh)
            for cell_idx, area in area_contrib.items():
                contributions[cell_idx] += area
        
        return contributions
    
    def compute_total_gradient_finite_difference(self, eps: float = 1e-6) -> np.ndarray:
        """
        Compute total gradient contribution from all Steiner trees.
        
        Args:
            eps: Perturbation size for finite differences
            
        Returns:
            Gradient array of shape (n_variable_points,)
        """
        gradient = np.zeros(len(self.partition.variable_points))
        
        for tp in self.triple_points:
            gradient += tp.compute_gradients_finite_difference(eps)
        
        return gradient
    
    def get_boundary_triple_points(self, tol: float = 1e-3) -> List[TriplePoint]:
        """
        Find triple points with Steiner point near triangle boundary.
        
        These may need topology switches (expansion to adjacent triangles).
        
        Args:
            tol: Distance tolerance
            
        Returns:
            List of TriplePoint objects near boundaries
        """
        boundary_tps = []
        for tp in self.triple_points:
            if tp.is_on_triangle_boundary(tol):
                boundary_tps.append(tp)
        
        return boundary_tps
    
    def update_after_lambda_change(self):
        """
        Recompute all Steiner points after λ parameters change.
        
        Should be called after optimization updates λ values.
        """
        for tp in self.triple_points:
            tp.steiner_point = None
            tp.compute_steiner_point()

