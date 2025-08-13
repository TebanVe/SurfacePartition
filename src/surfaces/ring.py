import numpy as np
from typing import Optional

try:
	from ..logging_config import get_logger
	from ..core.tri_mesh import TriMesh
	from ..ring_mesh import RingMesh
except Exception:
	import sys, os
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
	from logging_config import get_logger
	from core.tri_mesh import TriMesh
	from ring_mesh import RingMesh


class RingMeshProvider:
	"""
	Surface provider for a planar annulus. Builds a TriMesh from ring parameters.
	"""
	def __init__(self, n_radial: int, n_angular: int, r_inner: float, r_outer: float):
		self.logger = get_logger(__name__)
		self.n_radial = n_radial
		self.n_angular = n_angular
		self.r_inner = r_inner
		self.r_outer = r_outer

	def build(self) -> TriMesh:
		# Reuse existing ring vertex/face generator
		ring = RingMesh(self.n_radial, self.n_angular, self.r_inner, self.r_outer)
		# Do not reuse ring matrices; TriMesh will assemble its own
		vertices = ring.vertices
		faces = ring.faces
		return TriMesh(vertices, faces)

	def theoretical_total_area(self) -> float:
		return float(np.pi * (self.r_outer ** 2 - self.r_inner ** 2)) 