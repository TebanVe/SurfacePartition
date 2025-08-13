import os
import h5py
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Add src to path if needed
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))

from logging_config import get_logger


class ContourAnalyzer:
    """
    Analyze and visualize contours for ring partition results in R^2.

    This follows the paper's approach (see manifold_partition.md, eq. (5.1)):
    - Compute indicator functions via winner-takes-all on densities
    - Extract 0.5 level-set segments per region across mesh triangles
    """

    def __init__(self, result_path: str, logger=None):
        self.result_path = Path(result_path)
        self.logger = logger or get_logger(__name__)

        self.x: Optional[np.ndarray] = None
        self.vertices: Optional[np.ndarray] = None  # shape (N, 2)
        self.faces: Optional[np.ndarray] = None     # shape (T, 3)
        self.densities: Optional[np.ndarray] = None  # shape (N, n_partitions)
        self.level: float = 0.5

    def load_results(self, use_initial_condition: bool = False) -> None:
        """
        Load solution and mesh from .h5 file.

        Args:
            use_initial_condition: If True, load x0 instead of x_opt
        """
        if not self.result_path.exists() or not self.result_path.is_file():
            raise FileNotFoundError(f"Solution file not found: {self.result_path}")
        if self.result_path.suffix.lower() != ".h5":
            raise ValueError(f"Expected .h5 solution file, got: {self.result_path}")

        dataset = 'x0' if use_initial_condition else 'x_opt'
        with h5py.File(self.result_path, 'r') as f:
            if dataset not in f:
                raise ValueError(f"Dataset '{dataset}' not found in {self.result_path}")
            if 'vertices' not in f or 'faces' not in f:
                raise ValueError("Solution file must contain 'vertices' and 'faces' datasets")

            self.x = f[dataset][:]
            self.vertices = f['vertices'][:]
            self.faces = f['faces'][:]

        if self.vertices.ndim != 2 or self.vertices.shape[1] != 2:
            raise ValueError(f"Vertices must be (N,2); got {self.vertices.shape}")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(f"Faces must be (T,3); got {self.faces.shape}")

        n_vertices = self.vertices.shape[0]
        if self.x.shape[0] % n_vertices != 0:
            raise ValueError(
                f"Solution length {self.x.shape[0]} not divisible by n_vertices {n_vertices}"
            )
        n_partitions = self.x.shape[0] // n_vertices
        self.densities = self.x.reshape(n_vertices, n_partitions)

        self.logger.info(
            f"Loaded {'x0' if use_initial_condition else 'x_opt'}: "
            f"{n_vertices} vertices, {n_partitions} partitions"
        )

    def compute_indicator_functions(self) -> np.ndarray:
        """
        Compute indicator functions chi via winner-takes-all on densities.

        Returns:
            chi: (N, n_partitions) binary matrix
        """
        if self.densities is None:
            raise ValueError("Call load_results() before compute_indicator_functions()")

        n_vertices, n_partitions = self.densities.shape
        chi = np.zeros_like(self.densities)
        max_indices = np.argmax(self.densities, axis=1)
        chi[np.arange(n_vertices), max_indices] = 1.0
        return chi

    def _find_triangle_level_segments(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                                      d1: float, d2: float, d3: float, level: float) -> List[np.ndarray]:
        """
        Find up to one segment of the level-set within a triangle for a scalar field.

        Returns a list with either 0 or 1 segment; each segment is (2, 2).
        """
        points = []
        # Edge (p1, p2)
        if (d1 > level) != (d2 > level):
            t = (level - d1) / (d2 - d1)
            points.append(p1 + t * (p2 - p1))
        # Edge (p2, p3)
        if (d2 > level) != (d3 > level):
            t = (level - d2) / (d3 - d2)
            points.append(p2 + t * (p3 - p2))
        # Edge (p3, p1)
        if (d3 > level) != (d1 > level):
            t = (level - d3) / (d1 - d3)
            points.append(p3 + t * (p1 - p3))

        if len(points) == 2:
            return [np.vstack(points)]  # shape (2, 2)
        return []

    def extract_contours(self, level: float = 0.5) -> Dict[int, List[np.ndarray]]:
        """
        Extract contour segments per region using indicator functions at a given level.

        Args:
            level: level-set threshold (default 0.5)
        Returns:
            Dict region_index -> list of segments (each segment shape (2, 2))
        """
        if self.densities is None:
            raise ValueError("Call load_results() before extract_contours()")

        self.level = level
        chi = self.compute_indicator_functions()
        n_regions = chi.shape[1]

        contours: Dict[int, List[np.ndarray]] = {i: [] for i in range(n_regions)}

        for region_idx in range(n_regions):
            chi_region = chi[:, region_idx]
            segs: List[np.ndarray] = []

            for face in self.faces:
                v1, v2, v3 = map(int, face)
                d1, d2, d3 = chi_region[v1], chi_region[v2], chi_region[v3]

                # Only if triangle is cut by the level set
                if (d1 > level) != (d2 > level) or (d2 > level) != (d3 > level) or (d3 > level) != (d1 > level):
                    p1 = self.vertices[v1]
                    p2 = self.vertices[v2]
                    p3 = self.vertices[v3]
                    segs.extend(self._find_triangle_level_segments(p1, p2, p3, d1, d2, d3, level))

            contours[region_idx] = segs
            self.logger.info(f"Region {region_idx}: extracted {len(segs)} contour segments at level {level}")

        return contours

    def label_triangles_from_indicator(self) -> np.ndarray:
        """
        Assign a region label to each triangle via majority vote of its 3 vertices' labels.

        Returns:
            triangle_labels: (T,) integer labels in [0, n_partitions-1]
        """
        if self.densities is None:
            raise ValueError("Call load_results() before labeling triangles")

        vertex_labels = np.argmax(self.densities, axis=1)
        T = self.faces.shape[0]
        labels = np.zeros(T, dtype=int)
        for t, (v1, v2, v3) in enumerate(self.faces.astype(int)):
            votes = [vertex_labels[v1], vertex_labels[v2], vertex_labels[v3]]
            # Majority vote; if tie, pick the first max
            counts = np.bincount(votes, minlength=self.densities.shape[1])
            labels[t] = int(np.argmax(counts))
        return labels

    def stitch_segments_to_polylines(self, segments: List[np.ndarray], tol: float = 1e-8) -> List[np.ndarray]:
        """
        Greedy stitching of small line segments into ordered polylines by connecting
        endpoints within a tolerance. Returns list of polylines (M_i, 2).
        """
        if not segments:
            return []

        remaining = [seg.copy() for seg in segments]
        polylines: List[np.ndarray] = []

        while remaining:
            # Start a new polyline with one segment
            poly = remaining.pop()
            start, end = poly[0], poly[1]

            extended = True
            while extended:
                extended = False
                for i in range(len(remaining)):
                    s = remaining[i]
                    s0, s1 = s[0], s[1]
                    if np.linalg.norm(end - s0) < tol:
                        # append forward
                        poly = np.vstack([poly, s1])
                        end = s1
                        remaining.pop(i)
                        extended = True
                        break
                    if np.linalg.norm(end - s1) < tol:
                        # append reversed
                        poly = np.vstack([poly, s0])
                        end = s0
                        remaining.pop(i)
                        extended = True
                        break
                    if np.linalg.norm(start - s1) < tol:
                        # prepend forward
                        poly = np.vstack([s0, poly])
                        start = s0
                        remaining.pop(i)
                        extended = True
                        break
                    if np.linalg.norm(start - s0) < tol:
                        # prepend reversed
                        poly = np.vstack([s1, poly])
                        start = s1
                        remaining.pop(i)
                        extended = True
                        break

            polylines.append(poly)

        return polylines 