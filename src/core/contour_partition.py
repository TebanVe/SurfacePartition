"""
Contour partition data structures for perimeter refinement optimization.

This module implements Section 5 of the paper "Partitions of Minimal Length on Manifolds"
by Bogosel and Oudet. It provides data structures for representing partition contours
as variable points on mesh edges, enabling direct perimeter optimization.

Key classes:
- VariablePoint: Point on a mesh edge parameterized by λ ∈ [0,1]
- CellContour: Ordered contour segments forming one partition cell
- PartitionContour: Complete partition with global variable point management
"""

import numpy as np
import h5py
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass

try:
    from ..logging_config import get_logger
    from .tri_mesh import TriMesh
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from logging_config import get_logger
    from core.tri_mesh import TriMesh


@dataclass
class VariablePoint:
    """
    A point on a mesh edge parameterized by λ ∈ [0,1].
    
    Position: x = λ * v_start + (1-λ) * v_end
    
    Attributes:
        edge: Tuple of (vertex_idx_start, vertex_idx_end)
        lambda_param: Parameter value in [0, 1]
        global_idx: Index in the global variable vector
        adjacent_cells: Set of cell indices that use this point
    """
    edge: Tuple[int, int]
    lambda_param: float
    global_idx: int
    adjacent_cells: Set[int]
    
    def evaluate(self, vertices: np.ndarray) -> np.ndarray:
        """Compute actual 3D/2D coordinates given lambda."""
        v_start = vertices[self.edge[0]]
        v_end = vertices[self.edge[1]]
        return self.lambda_param * v_start + (1 - self.lambda_param) * v_end
    
    def on_boundary(self, tol: float = 1e-3) -> bool:
        """Check if point is near edge endpoints (topology switch condition)."""
        return self.lambda_param < tol or self.lambda_param > (1 - tol)


class CellContour:
    """
    Represents the contour of one partition cell.
    
    The contour is stored as an ordered list of variable point indices,
    where consecutive points form segments. The contour may consist of
    multiple connected components in complex topologies.
    
    Attributes:
        cell_idx: Index of this cell in the partition
        contour_points: Ordered list of global variable point indices
        connected_components: List of lists, each a closed contour loop
    """
    
    def __init__(self, cell_idx: int):
        self.cell_idx = cell_idx
        self.contour_points: List[int] = []
        self.connected_components: List[List[int]] = []
    
    def add_point(self, var_point_idx: int):
        """Add a variable point to the contour."""
        self.contour_points.append(var_point_idx)
    
    def organize_into_components(self, partition: 'PartitionContour'):
        """
        Organize contour points into connected components (closed loops).
        This is needed for cells with multiple boundaries or complex topology.
        """
        if not self.contour_points:
            return
        
        # For now, assume single connected component (most common case)
        # TODO: Implement proper connected component analysis if needed
        self.connected_components = [self.contour_points.copy()]
    
    def get_segments(self) -> List[Tuple[int, int]]:
        """
        Get list of segments as pairs of consecutive variable point indices.
        For closed contours, includes segment from last back to first point.
        """
        segments = []
        for component in self.connected_components:
            if len(component) < 2:
                continue
            # Consecutive pairs
            for i in range(len(component) - 1):
                segments.append((component[i], component[i+1]))
            # Close the loop
            if len(component) > 2:
                segments.append((component[-1], component[0]))
        return segments


class PartitionContour:
    """
    Global partition representation with variable points on mesh edges.
    
    This is the main data structure for Section 5 optimization. It manages:
    - All variable points across the partition
    - Contours for each cell
    - Topology information (which edges form which cells)
    - Conversion to/from indicator functions
    
    Attributes:
        mesh: The underlying TriMesh
        n_cells: Number of partition cells
        variable_points: List of all VariablePoint objects
        cells: List of CellContour objects
        indicator_functions: (N, n_cells) array of φ_i from equation (5.1)
        edge_to_varpoint: Map from edge tuple to variable point index
        triple_points: List of detected triple points (computed on demand)
    """
    
    def __init__(self, mesh: TriMesh, indicator_functions: np.ndarray):
        """
        Initialize partition contours from indicator functions φ_i.
        
        Args:
            mesh: TriMesh object
            indicator_functions: (N, n_cells) binary array from winner-takes-all
        """
        self.mesh = mesh
        self.logger = get_logger(__name__)
        self.indicator_functions = indicator_functions
        self.n_cells = indicator_functions.shape[1]
        
        # Global data structures
        self.variable_points: List[VariablePoint] = []
        self.cells: List[CellContour] = [CellContour(i) for i in range(self.n_cells)]
        self.edge_to_varpoint: Dict[Tuple[int, int], int] = {}
        self.triple_points: Optional[List] = None  # Computed on demand
        
        # Build the contour representation
        self._initialize_from_indicators()
        
        self.logger.info(f"Initialized PartitionContour: {len(self.variable_points)} variable points, "
                        f"{self.n_cells} cells")
    
    def _initialize_from_indicators(self):
        """
        Extract contours from indicator functions by finding mesh edges
        that cross cell boundaries (where φ_i changes from 0 to 1).
        """
        vertex_labels = np.argmax(self.indicator_functions, axis=1)
        
        # Iterate over all triangles to find boundary edges
        for tri_idx, face in enumerate(self.mesh.faces):
            v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
            label1, label2, label3 = vertex_labels[v1], vertex_labels[v2], vertex_labels[v3]
            
            # Check each edge of the triangle
            edges = [(v1, v2), (v2, v3), (v3, v1)]
            labels = [(label1, label2), (label2, label3), (label3, label1)]
            
            for edge, (lab_a, lab_b) in zip(edges, labels):
                if lab_a != lab_b:
                    # This edge crosses a boundary
                    # Normalize edge representation (smaller index first)
                    normalized_edge = tuple(sorted(edge))
                    
                    if normalized_edge not in self.edge_to_varpoint:
                        # Create new variable point at midpoint (λ = 0.5)
                        var_point = VariablePoint(
                            edge=normalized_edge,
                            lambda_param=0.5,
                            global_idx=len(self.variable_points),
                            adjacent_cells={lab_a, lab_b}
                        )
                        self.variable_points.append(var_point)
                        self.edge_to_varpoint[normalized_edge] = var_point.global_idx
                        
                        # Add to both adjacent cells' contours
                        self.cells[lab_a].add_point(var_point.global_idx)
                        self.cells[lab_b].add_point(var_point.global_idx)
                    else:
                        # Update adjacent cells if this edge appears in multiple triangles
                        var_idx = self.edge_to_varpoint[normalized_edge]
                        self.variable_points[var_idx].adjacent_cells.update([lab_a, lab_b])
        
        # Organize each cell's points into connected components
        for cell in self.cells:
            cell.organize_into_components(self)
    
    def get_variable_vector(self) -> np.ndarray:
        """
        Return current λ parameters as optimization vector.
        
        Returns:
            Array of shape (n_variable_points,) with λ values
        """
        return np.array([vp.lambda_param for vp in self.variable_points])
    
    def set_variable_vector(self, lambda_vec: np.ndarray):
        """
        Update all λ parameters from optimization vector.
        
        Args:
            lambda_vec: Array of shape (n_variable_points,) with new λ values
        """
        if len(lambda_vec) != len(self.variable_points):
            raise ValueError(f"Lambda vector size {len(lambda_vec)} doesn't match "
                           f"number of variable points {len(self.variable_points)}")
        
        for i, lam in enumerate(lambda_vec):
            self.variable_points[i].lambda_param = float(np.clip(lam, 0.0, 1.0))
    
    def evaluate_variable_point(self, var_point_idx: int) -> np.ndarray:
        """Get 3D/2D coordinates of a variable point."""
        return self.variable_points[var_point_idx].evaluate(self.mesh.vertices)
    
    def to_visualization_format(self) -> Dict[int, List[np.ndarray]]:
        """
        Export refined contours in the same format as ContourAnalyzer.extract_contours().
        
        This allows refined contours to be visualized using existing plot functions.
        
        Returns:
            Dict[region_idx] -> List[segment arrays (2, D)]
            where D is 2 or 3 depending on mesh dimension
        """
        contours_dict = {i: [] for i in range(self.n_cells)}
        
        for cell in self.cells:
            segments = cell.get_segments()
            for seg_start_idx, seg_end_idx in segments:
                p_start = self.evaluate_variable_point(seg_start_idx)
                p_end = self.evaluate_variable_point(seg_end_idx)
                segment = np.vstack([p_start, p_end])
                contours_dict[cell.cell_idx].append(segment)
        
        self.logger.info(f"Converted to visualization format: "
                        f"{sum(len(segs) for segs in contours_dict.values())} total segments")
        
        return contours_dict
    
    def save_refined_contours(self, output_path: str, 
                             perimeter: float,
                             areas: List[float],
                             optimization_info: Dict):
        """
        Save refined contours to HDF5 for visualization and analysis.
        
        Stores:
        - Optimized λ parameters
        - Evaluated contour segments (for visualization)
        - Triple point information
        - Optimization metadata (perimeter, areas, convergence info)
        
        Args:
            output_path: Path to HDF5 file
            perimeter: Final optimized total perimeter
            areas: List of cell areas
            optimization_info: Dict with optimization metadata
        """
        with h5py.File(output_path, 'w') as f:
            # Global metadata
            f.attrs['n_cells'] = self.n_cells
            f.attrs['n_variable_points'] = len(self.variable_points)
            f.attrs['final_perimeter'] = float(perimeter)
            f.attrs['target_area'] = float(optimization_info.get('target_area', 0.0))
            f.attrs['optimization_success'] = bool(optimization_info.get('success', False))
            f.attrs['n_iterations'] = int(optimization_info.get('n_iterations', 0))
            f.attrs['mesh_dimension'] = int(self.mesh.dim)
            
            # Save λ parameters
            lambda_vec = self.get_variable_vector()
            f.create_dataset('lambda_parameters', data=lambda_vec)
            
            # Save variable point metadata
            vp_grp = f.create_group('variable_points')
            for i, vp in enumerate(self.variable_points):
                vp_subgrp = vp_grp.create_group(f'vp_{i}')
                vp_subgrp.attrs['edge_start'] = vp.edge[0]
                vp_subgrp.attrs['edge_end'] = vp.edge[1]
                vp_subgrp.attrs['lambda'] = vp.lambda_param
                vp_subgrp.attrs['adjacent_cells'] = list(vp.adjacent_cells)
            
            # Save evaluated contours in visualization format
            viz_contours = self.to_visualization_format()
            for cell_idx, segments in viz_contours.items():
                grp = f.create_group(f'cell_{cell_idx}')
                grp.attrs['n_segments'] = len(segments)
                grp.attrs['area'] = float(areas[cell_idx]) if cell_idx < len(areas) else 0.0
                for seg_idx, seg in enumerate(segments):
                    grp.create_dataset(f'segment_{seg_idx}', data=seg)
            
            # Save triple points info if available
            if self.triple_points is not None and len(self.triple_points) > 0:
                tp_grp = f.create_group('triple_points')
                tp_grp.attrs['n_triple_points'] = len(self.triple_points)
                # Triple point details will be filled in by steiner_handler module
        
        self.logger.info(f"Saved refined contours to: {output_path}")
    
    @staticmethod
    def load_refined_contours(input_path: str) -> Dict[int, List[np.ndarray]]:
        """
        Load refined contours from HDF5 file in visualization format.
        
        Args:
            input_path: Path to HDF5 file created by save_refined_contours()
            
        Returns:
            Dict[cell_idx] -> List[segment arrays (2, D)]
        """
        with h5py.File(input_path, 'r') as f:
            n_cells = int(f.attrs['n_cells'])
            contours = {}
            
            for cell_idx in range(n_cells):
                grp = f[f'cell_{cell_idx}']
                n_segments = int(grp.attrs['n_segments'])
                segments = []
                for seg_idx in range(n_segments):
                    seg = grp[f'segment_{seg_idx}'][:]
                    segments.append(seg)
                contours[cell_idx] = segments
        
        return contours
    
    def identify_triple_points(self) -> List[Tuple[int, List[int]]]:
        """
        Identify triangles where three different regions meet (triple points).
        
        These are triangles with three variable points from three different cells,
        creating small void spaces that need Steiner tree treatment (Section 5).
        
        Returns:
            List of (triangle_idx, [var_point_idx1, var_point_idx2, var_point_idx3])
        """
        triple_points = []
        vertex_labels = np.argmax(self.indicator_functions, axis=1)
        
        for tri_idx, face in enumerate(self.mesh.faces):
            v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
            labels = {vertex_labels[v1], vertex_labels[v2], vertex_labels[v3]}
            
            # Triple point: all 3 vertices belong to different regions
            if len(labels) == 3:
                # Find the 3 variable points on this triangle's edges
                edges = [
                    tuple(sorted([v1, v2])),
                    tuple(sorted([v2, v3])),
                    tuple(sorted([v3, v1]))
                ]
                var_points = []
                for edge in edges:
                    if edge in self.edge_to_varpoint:
                        var_points.append(self.edge_to_varpoint[edge])
                
                if len(var_points) == 3:
                    triple_points.append((tri_idx, var_points))
        
        self.logger.info(f"Identified {len(triple_points)} triple points")
        return triple_points
    
    def get_boundary_variable_points(self, tol: float = 1e-3) -> List[int]:
        """
        Find variable points near edge endpoints (candidates for topology switch).
        
        Args:
            tol: Threshold for considering a point at boundary
            
        Returns:
            List of variable point indices with λ < tol or λ > 1-tol
        """
        boundary_points = []
        for vp in self.variable_points:
            if vp.on_boundary(tol):
                boundary_points.append(vp.global_idx)
        
        return boundary_points

