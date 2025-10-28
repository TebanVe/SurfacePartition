"""
Area computation and gradient calculation for partition cells on triangulated surfaces.

This module implements the area computation logic from Section 5 of the paper:
"Computation of the areas of the cells". It handles:
- Full mesh triangles completely inside a partition cell
- Partial mesh triangles cut by contour lines (depends on λ parameters)
- Analytical gradients ∂Area/∂λ for optimization

TERMINOLOGY (following paper Section 5):
- "cell": Partition region (what we optimize for equal areas)
- "triangle": Mesh triangle element (computational discretization)
- "edge": Mesh triangle edge (computational discretization)

For each mesh triangle, we determine its contribution to each partition cell's area
based on the indicator functions φ_i and the current variable point positions.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional

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


class AreaCalculator:
    """
    Computes partition cell areas and gradients for perimeter optimization.
    
    For each partition cell, the area is sum of:
    1. Full mesh triangles where all 3 vertices belong to the cell
    2. Partial mesh triangles where 2 vertices belong (trapezoid/triangle portion)
    3. Partial mesh triangles where 1 vertex belongs (small triangle portion)
    
    The partial triangle areas depend on λ parameters of variable points,
    and we provide analytical gradients for optimization.
    
    Attributes:
        mesh: The underlying TriMesh
        partition: PartitionContour with variable points
        vertex_labels: (N,) array mapping vertices to cell indices
        triangle_areas: (T,) array of mesh triangle areas (cached)
    """
    
    def __init__(self, mesh: TriMesh, partition: PartitionContour):
        """
        Initialize area calculator.
        
        Args:
            mesh: TriMesh object
            partition: PartitionContour with indicator functions and variable points
        """
        self.mesh = mesh
        self.partition = partition
        self.logger = get_logger(__name__)
        
        # Cache vertex labels from indicator functions
        self.vertex_labels = np.argmax(partition.indicator_functions, axis=1)
        
        # Cache triangle areas for efficiency
        self.triangle_areas = mesh.triangle_areas
        
        self.logger.info(f"Initialized AreaCalculator for {partition.n_cells} cells, "
                        f"{mesh.faces.shape[0]} triangles")
    
    def compute_all_cell_areas(self, lambda_vec: np.ndarray) -> np.ndarray:
        """
        Compute areas of all cells given current λ parameters.
        
        Args:
            lambda_vec: Current variable point parameters
            
        Returns:
            Array of shape (n_cells,) with cell areas
        """
        self.partition.set_variable_vector(lambda_vec)
        areas = np.zeros(self.partition.n_cells)
        
        for cell_idx in range(self.partition.n_cells):
            areas[cell_idx] = self.compute_cell_area(cell_idx, lambda_vec)
        
        return areas
    
    def compute_cell_area(self, cell_idx: int, lambda_vec: np.ndarray) -> float:
        """
        Compute total area of one cell.
        
        Args:
            cell_idx: Index of the cell
            lambda_vec: Current variable point parameters
            
        Returns:
            Total area of the cell
        """
        self.partition.set_variable_vector(lambda_vec)
        total_area = 0.0
        
        # Iterate over all mesh triangles
        for tri_idx in range(self.mesh.faces.shape[0]):
            area_contrib, _ = self._triangle_contribution(tri_idx, cell_idx, lambda_vec)
            total_area += area_contrib
        
        return total_area
    
    def compute_area_gradient(self, cell_idx: int, lambda_vec: np.ndarray) -> np.ndarray:
        """
        Compute gradient ∂(Area_i)/∂λ for all variable points.
        
        Args:
            cell_idx: Index of the cell
            lambda_vec: Current variable point parameters
            
        Returns:
            Gradient array of shape (n_variable_points,)
        """
        self.partition.set_variable_vector(lambda_vec)
        gradient = np.zeros(len(lambda_vec))
        
        # Accumulate gradients from all mesh triangles
        for tri_idx in range(self.mesh.faces.shape[0]):
            _, grad_contrib = self._triangle_contribution(tri_idx, cell_idx, lambda_vec)
            gradient += grad_contrib
        
        return gradient
    
    def _triangle_contribution(self, tri_idx: int, cell_idx: int, 
                               lambda_vec: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute area contribution and gradient from one mesh triangle to one partition cell.
        
        Returns:
            (area_contribution, gradient_contribution)
            where gradient_contribution is shape (n_variable_points,)
        """
        face = self.mesh.faces[tri_idx]
        v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
        labels = [self.vertex_labels[v1], self.vertex_labels[v2], self.vertex_labels[v3]]
        
        # Count how many vertices belong to this partition cell
        n_inside = sum(1 for lab in labels if lab == cell_idx)
        
        gradient = np.zeros(len(lambda_vec))
        
        if n_inside == 3:
            # Case 1: Mesh triangle fully inside partition cell
            return self.triangle_areas[tri_idx], gradient
        
        elif n_inside == 0:
            # Case 4: Mesh triangle fully outside partition cell
            return 0.0, gradient
        
        elif n_inside == 2:
            # Case 2: Two vertices inside, one outside
            return self._partial_area_two_inside(tri_idx, cell_idx, v1, v2, v3, labels)
        
        else:  # n_inside == 1
            # Case 3: One vertex inside, two outside
            return self._partial_area_one_inside(tri_idx, cell_idx, v1, v2, v3, labels)
    
    def _partial_area_two_inside(self, tri_idx: int, cell_idx: int,
                                 v1: int, v2: int, v3: int,
                                 labels: List[int]) -> Tuple[float, np.ndarray]:
        """
        Compute area when 2 vertices are inside the cell (trapezoid or triangle).
        
        The contour cuts the triangle, leaving a trapezoid/triangle portion inside.
        Area depends on λ parameters of the two edges connecting to the outside vertex.
        """
        # Identify which vertices are inside and which is outside
        vertices = [v1, v2, v3]
        inside_mask = [lab == cell_idx for lab in labels]
        
        # Find the outside vertex and the two inside vertices
        outside_idx = None
        inside_indices = []
        for i, is_inside in enumerate(inside_mask):
            if is_inside:
                inside_indices.append(i)
            else:
                outside_idx = i
        
        if outside_idx is None or len(inside_indices) != 2:
            # Shouldn't happen, but return zero if it does
            return 0.0, np.zeros(len(self.partition.variable_points))
        
        v_out = vertices[outside_idx]
        v_in1 = vertices[inside_indices[0]]
        v_in2 = vertices[inside_indices[1]]
        
        # Find variable points on edges (v_out, v_in1) and (v_out, v_in2)
        edge1 = tuple(sorted([v_out, v_in1]))
        edge2 = tuple(sorted([v_out, v_in2]))
        
        if edge1 not in self.partition.edge_to_varpoint or edge2 not in self.partition.edge_to_varpoint:
            # No variable points on these edges (shouldn't happen)
            return 0.0, np.zeros(len(self.partition.variable_points))
        
        vp_idx1 = self.partition.edge_to_varpoint[edge1]
        vp_idx2 = self.partition.edge_to_varpoint[edge2]
        
        lambda1 = self.partition.variable_points[vp_idx1].lambda_param
        lambda2 = self.partition.variable_points[vp_idx2].lambda_param
        
        # Compute positions of variable points
        p_out = self.mesh.vertices[v_out]
        p_in1 = self.mesh.vertices[v_in1]
        p_in2 = self.mesh.vertices[v_in2]
        
        # Variable points on edges
        # Note: Need to respect edge orientation
        if edge1[0] == v_out:
            p_cut1 = lambda1 * p_out + (1 - lambda1) * p_in1
        else:
            p_cut1 = (1 - lambda1) * p_out + lambda1 * p_in1
        
        if edge2[0] == v_out:
            p_cut2 = lambda2 * p_out + (1 - lambda2) * p_in2
        else:
            p_cut2 = (1 - lambda2) * p_out + lambda2 * p_in2
        
        # The region inside the cell is a quadrilateral: p_in1, p_cut1, p_cut2, p_in2
        # Compute area using cross product formula
        area = self._quadrilateral_area(p_in1, p_cut1, p_cut2, p_in2)
        
        # Compute gradient (simplified - more accurate implementation would use chain rule)
        # For now, use finite differences as suggested by paper for Steiner points
        gradient = np.zeros(len(self.partition.variable_points))
        
        # Finite difference for affected variable points
        eps = 1e-7
        
        # Perturb first variable point
        lambda1_perturbed = lambda1 + eps
        if edge1[0] == v_out:
            p_cut1_perturbed = lambda1_perturbed * p_out + (1 - lambda1_perturbed) * p_in1
        else:
            p_cut1_perturbed = (1 - lambda1_perturbed) * p_out + lambda1_perturbed * p_in1
        area_perturbed1 = self._quadrilateral_area(p_in1, p_cut1_perturbed, p_cut2, p_in2)
        gradient[vp_idx1] = (area_perturbed1 - area) / eps
        
        # Perturb second variable point
        lambda2_perturbed = lambda2 + eps
        if edge2[0] == v_out:
            p_cut2_perturbed = lambda2_perturbed * p_out + (1 - lambda2_perturbed) * p_in2
        else:
            p_cut2_perturbed = (1 - lambda2_perturbed) * p_out + lambda2_perturbed * p_in2
        area_perturbed2 = self._quadrilateral_area(p_in1, p_cut1, p_cut2_perturbed, p_in2)
        gradient[vp_idx2] = (area_perturbed2 - area) / eps
        
        return area, gradient
    
    def _partial_area_one_inside(self, tri_idx: int, cell_idx: int,
                                 v1: int, v2: int, v3: int,
                                 labels: List[int]) -> Tuple[float, np.ndarray]:
        """
        Compute area when 1 vertex is inside the cell (small triangle).
        
        The contour cuts the triangle, leaving a small triangular portion inside.
        Area depends on λ parameters of the two edges connecting to the inside vertex.
        """
        # Identify which vertex is inside
        vertices = [v1, v2, v3]
        inside_mask = [lab == cell_idx for lab in labels]
        
        inside_idx = None
        outside_indices = []
        for i, is_inside in enumerate(inside_mask):
            if is_inside:
                inside_idx = i
            else:
                outside_indices.append(i)
        
        if inside_idx is None or len(outside_indices) != 2:
            return 0.0, np.zeros(len(self.partition.variable_points))
        
        v_in = vertices[inside_idx]
        v_out1 = vertices[outside_indices[0]]
        v_out2 = vertices[outside_indices[1]]
        
        # Find variable points on edges (v_in, v_out1) and (v_in, v_out2)
        edge1 = tuple(sorted([v_in, v_out1]))
        edge2 = tuple(sorted([v_in, v_out2]))
        
        if edge1 not in self.partition.edge_to_varpoint or edge2 not in self.partition.edge_to_varpoint:
            return 0.0, np.zeros(len(self.partition.variable_points))
        
        vp_idx1 = self.partition.edge_to_varpoint[edge1]
        vp_idx2 = self.partition.edge_to_varpoint[edge2]
        
        lambda1 = self.partition.variable_points[vp_idx1].lambda_param
        lambda2 = self.partition.variable_points[vp_idx2].lambda_param
        
        # Compute positions
        p_in = self.mesh.vertices[v_in]
        p_out1 = self.mesh.vertices[v_out1]
        p_out2 = self.mesh.vertices[v_out2]
        
        # Variable points on edges
        if edge1[0] == v_in:
            p_cut1 = lambda1 * p_in + (1 - lambda1) * p_out1
        else:
            p_cut1 = (1 - lambda1) * p_in + lambda1 * p_out1
        
        if edge2[0] == v_in:
            p_cut2 = lambda2 * p_in + (1 - lambda2) * p_out2
        else:
            p_cut2 = (1 - lambda2) * p_in + lambda2 * p_out2
        
        # The region inside is triangle: p_in, p_cut1, p_cut2
        area = self._triangle_area_3d(p_in, p_cut1, p_cut2)
        
        # Gradient via finite differences
        gradient = np.zeros(len(self.partition.variable_points))
        
        eps = 1e-7
        
        # Perturb first variable point
        lambda1_perturbed = lambda1 + eps
        if edge1[0] == v_in:
            p_cut1_perturbed = lambda1_perturbed * p_in + (1 - lambda1_perturbed) * p_out1
        else:
            p_cut1_perturbed = (1 - lambda1_perturbed) * p_in + lambda1_perturbed * p_out1
        area_perturbed1 = self._triangle_area_3d(p_in, p_cut1_perturbed, p_cut2)
        gradient[vp_idx1] = (area_perturbed1 - area) / eps
        
        # Perturb second variable point
        lambda2_perturbed = lambda2 + eps
        if edge2[0] == v_in:
            p_cut2_perturbed = lambda2_perturbed * p_in + (1 - lambda2_perturbed) * p_out2
        else:
            p_cut2_perturbed = (1 - lambda2_perturbed) * p_in + lambda2_perturbed * p_out2
        area_perturbed2 = self._triangle_area_3d(p_in, p_cut1, p_cut2_perturbed)
        gradient[vp_idx2] = (area_perturbed2 - area) / eps
        
        return area, gradient
    
    def _triangle_area_3d(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Compute area of triangle in 2D or 3D using cross product.
        
        Works for both planar (2D) and embedded (3D) triangles.
        """
        if len(p1) == 2:
            # 2D case
            v1 = p2 - p1
            v2 = p3 - p1
            # Cross product in 2D gives scalar
            return 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
        else:
            # 3D case
            v1 = p2 - p1
            v2 = p3 - p1
            cross = np.cross(v1, v2)
            return 0.5 * np.linalg.norm(cross)
    
    def _quadrilateral_area(self, p1: np.ndarray, p2: np.ndarray, 
                           p3: np.ndarray, p4: np.ndarray) -> float:
        """
        Compute area of quadrilateral by splitting into two triangles.
        
        Quadrilateral vertices in order: p1, p2, p3, p4
        """
        # Split into triangles (p1, p2, p3) and (p1, p3, p4)
        area1 = self._triangle_area_3d(p1, p2, p3)
        area2 = self._triangle_area_3d(p1, p3, p4)
        return area1 + area2
    
    def compute_area_constraints(self, lambda_vec: np.ndarray, 
                                target_area: float) -> np.ndarray:
        """
        Compute area constraint violations: Area_i - target_area.
        
        Used in optimization to enforce equal-area constraints.
        Last cell area is not constrained (determined by others).
        
        Args:
            lambda_vec: Current variable point parameters
            target_area: Target area for each cell
            
        Returns:
            Array of shape (n_cells - 1,) with constraint violations
        """
        areas = self.compute_all_cell_areas(lambda_vec)
        return areas[:-1] - target_area
    
    def compute_area_jacobian(self, lambda_vec: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of area constraints: ∂(Area_i - target)/∂λ.
        
        Args:
            lambda_vec: Current variable point parameters
            
        Returns:
            Jacobian array of shape (n_cells - 1, n_variable_points)
        """
        n_constraints = self.partition.n_cells - 1
        n_vars = len(lambda_vec)
        jacobian = np.zeros((n_constraints, n_vars))
        
        for cell_idx in range(n_constraints):
            jacobian[cell_idx, :] = self.compute_area_gradient(cell_idx, lambda_vec)
        
        return jacobian

