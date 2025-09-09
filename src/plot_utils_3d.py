import numpy as np
from typing import Dict, List, Optional


def _faces_to_pyvista(faces: np.ndarray) -> np.ndarray:
	"""Convert (T,3) faces to PyVista flattened format [3,i,j,k, ...]."""
	if faces.ndim != 2 or faces.shape[1] != 3:
		raise ValueError(f"faces must be (T,3); got {faces.shape}")
	return np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64).ravel()


def _compute_triangle_centers_and_normals(vertices: np.ndarray, faces: np.ndarray):
	"""Compute triangle centers and unit normals directly from vertices/faces."""
	T = faces.shape[0]
	centers = np.zeros((T, vertices.shape[1]))
	normals = np.zeros((T, vertices.shape[1]))
	for ti, (i, j, k) in enumerate(faces.astype(int)):
		p1, p2, p3 = vertices[i], vertices[j], vertices[k]
		centers[ti] = (p1 + p2 + p3) / 3.0
		n = np.cross(p2 - p1, p3 - p1)
		nrm = np.linalg.norm(n)
		normals[ti] = n / nrm if nrm > 0 else np.array([0.0, 0.0, 1.0])
	return centers, normals


def plot_mesh_with_contours_pyvista(
	vertices: np.ndarray,
	faces: np.ndarray,
	contours: Optional[Dict[int, List[np.ndarray]]] = None,
	*,
	show_edges: bool = True,
	show_normals: bool = False,
	normal_scale: float = 0.1,
	normal_color: str = 'yellow',
	edge_color: str = 'white',
	opacity: float = 1.0,
	edge_line_width: float = 1.0,
	triangle_centers: Optional[np.ndarray] = None,
	normals: Optional[np.ndarray] = None,
	face_labels: Optional[np.ndarray] = None,
	show_scalar_bar: bool = False,
	palette: Optional[List[str]] = None,
	save_path: Optional[str] = None,
):
	"""
	Plot a 3D surface mesh with optional contours using PyVista.
	- vertices: (N,3)
	- faces: (T,3)
	- contours: dict region -> list of segments, each segment (2,3)
	"""
	try:
		import pyvista as pv  # type: ignore
	except Exception as e:
		raise ImportError("PyVista is required for 3D visualization. Please install 'pyvista'.") from e

	if vertices.ndim != 2 or vertices.shape[1] != 3:
		raise ValueError(f"vertices must be (N,3) for 3D plotting; got {vertices.shape}")

	faces_pv = _faces_to_pyvista(faces)
	mesh = pv.PolyData(vertices, faces_pv)
	plotter = pv.Plotter()

	# If categorical face colors provided, attach as cell_data and render with categories
	if face_labels is not None:
		labels = np.asarray(face_labels).astype(int)
		if labels.shape[0] != faces.shape[0]:
			raise ValueError("face_labels length must match number of faces")
		mesh.cell_data['region'] = labels
		# Default palette if not provided
		if palette is None:
			palette = [
				'#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5', '#70AD47',
				'#264478', '#9E480E', '#636363', '#997300', '#255E91', '#43682B'
			]
		plotter.add_mesh(
			mesh,
			scalars='region',
			categories=True,
			cmap=palette,
			show_scalar_bar=show_scalar_bar,
			opacity=opacity,
			show_edges=show_edges,
			edge_color=edge_color,
			line_width=edge_line_width,
		)
	else:
		plotter.add_mesh(
			mesh,
			color='lightgray',
			show_edges=show_edges,
			edge_color=edge_color,
			opacity=opacity,
			line_width=edge_line_width,
		)

	if show_normals:
		if triangle_centers is None or normals is None:
			triangle_centers, normals = _compute_triangle_centers_and_normals(vertices, faces)
		# Estimate arrow length from average neighbor spacing (simple heuristic)
		if triangle_centers.shape[0] > 1:
			# sample a few distances
			idx = min(10, triangle_centers.shape[0] - 1)
			dist = np.linalg.norm(triangle_centers[1:idx+1] - triangle_centers[:idx], axis=1)
			arrow_len = float(np.mean(dist)) * float(normal_scale)
		else:
			arrow_len = 0.1 * float(normal_scale)
		for c, n in zip(triangle_centers, normals):
			end = c + arrow_len * n
			arrow = pv.Arrow(start=c, direction=(end - c), scale=0.2)
			plotter.add_mesh(arrow, color=normal_color, opacity=0.8)

	# Add contours if provided (use strong colors to stand out over light fills)
	if contours is not None:
		color_list = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#17becf', '#e377c2', '#7f7f7f']
		for region_idx, segments in contours.items():
			color = color_list[region_idx % len(color_list)]
			for seg in segments:
				if seg.shape[1] != 3:
					raise ValueError("Contour segments must be (2,3) for 3D plotting")
				line = pv.Line(seg[0], seg[1])
				plotter.add_mesh(line, color=color, line_width=5)

	plotter.show()
	if save_path:
		import os as _os
		_outdir = _os.path.dirname(save_path)
		if _outdir:
			_os.makedirs(_outdir, exist_ok=True)
		plotter.screenshot(save_path)
