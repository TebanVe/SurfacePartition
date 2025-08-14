import numpy as np
from typing import Optional, Tuple

try:
	from ..logging_config import get_logger
	from ..core.tri_mesh import TriMesh
except Exception:
	import sys, os
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
	from logging_config import get_logger
	from core.tri_mesh import TriMesh


class RingMeshProvider:
	"""
	Surface provider for a planar annulus. Builds a TriMesh from ring parameters and
	provides naming metadata for orchestrators.
	"""
	def __init__(self, n_radial: int, n_angular: int, r_inner: float, r_outer: float,
				 n_radial_increment: int = 0, n_angular_increment: int = 0):
		self.logger = get_logger(__name__)
		self.n_radial = n_radial
		self.n_angular = n_angular
		self.r_inner = r_inner
		self.r_outer = r_outer
		self.init_n_radial = n_radial
		self.init_n_angular = n_angular
		self.incr_n_radial = n_radial_increment
		self.incr_n_angular = n_angular_increment

	# Surface identity and resolution labels
	def surface_name(self) -> str:
		return "ring"

	def resolution_labels(self) -> Tuple[str, str]:
		return ("nr", "na")

	def get_resolution(self) -> Tuple[int, int]:
		return (int(self.n_radial), int(self.n_angular))

	def set_resolution(self, n1: int, n2: int) -> None:
		self.n_radial = int(n1)
		self.n_angular = int(n2)

	def resolution_summary(self, refinement_levels: int) -> Tuple[str, str]:
		if refinement_levels > 1:
			final_nr = self.init_n_radial + (refinement_levels - 1) * self.incr_n_radial
			final_na = self.init_n_angular + (refinement_levels - 1) * self.incr_n_angular
			v1 = f"{self.init_n_radial}-{final_nr}_incr{self.incr_n_radial}"
			v2 = f"{self.init_n_angular}-{final_na}_incr{self.incr_n_angular}"
			return v1, v2
		else:
			return f"{self.init_n_radial}", f"{self.init_n_angular}"

	def _generate_vertices(self) -> np.ndarray:
		"""Generate ring vertices in polar coordinates (R2)."""
		r = np.linspace(self.r_inner, self.r_outer, self.n_radial)
		theta = np.linspace(0, 2 * np.pi, self.n_angular, endpoint=False)
		verts = []
		for radius in r:
			for angle in theta:
				x = radius * np.cos(angle)
				y = radius * np.sin(angle)
				verts.append([x, y])
		return np.array(verts)

	def _generate_triangles(self) -> np.ndarray:
		"""Generate counterclockwise triangles for the structured ring grid."""
		tris = []
		for i in range(self.n_radial - 1):
			for j in range(self.n_angular):
				current = i * self.n_angular + j
				next_radial = (i + 1) * self.n_angular + j
				next_angular = i * self.n_angular + ((j + 1) % self.n_angular)
				next_both = (i + 1) * self.n_angular + ((j + 1) % self.n_angular)
				# Triangle 1: [current, next_radial, next_angular]
				tris.append([current, next_radial, next_angular])
				# Triangle 2: [next_radial, next_both, next_angular]
				tris.append([next_radial, next_both, next_angular])
		return np.array(tris, dtype=int)

	def build(self) -> TriMesh:
		# Generate vertices/faces directly to avoid depending on RingMesh
		self.logger.info(f"Building ring TriMesh: nr={self.n_radial}, na={self.n_angular}, r_in={self.r_inner}, r_out={self.r_outer}")
		vertices = self._generate_vertices()
		faces = self._generate_triangles()
		return TriMesh(vertices, faces)

	def theoretical_total_area(self) -> float:
		return float(np.pi * (self.r_outer ** 2 - self.r_inner ** 2)) 