import numpy as np
from typing import Dict, Any, List, Tuple
import matplotlib.path as mpath
import matplotlib.tri as tri
import matplotlib.pyplot as plt


def compute_delta(densities: np.ndarray) -> np.ndarray:
    """
    For n=2, compute Δ = u1 - u2 per vertex.
    densities: (N, 2)
    """
    if densities.shape[1] != 2:
        raise ValueError("compute_delta expects densities with 2 columns (n=2)")
    return densities[:, 0] - densities[:, 1]


ess = 1e-12

def compute_confidence(densities: np.ndarray) -> np.ndarray:
    """m = max(u_i) per vertex."""
    return densities.max(axis=1)


def compute_entropy(densities: np.ndarray) -> np.ndarray:
    """Shannon entropy per vertex: H = -sum u_i log u_i."""
    clipped = np.clip(densities, ess, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def ambiguity_mask(delta: np.ndarray, tau: float = 0.1) -> np.ndarray:
    """Vertices where |Δ| < tau are considered ambiguous."""
    return np.abs(delta) < tau


def summarize_fields(densities: np.ndarray) -> Dict[str, Any]:
    """Return basic stats for diagnosing constant/ambiguous phases."""
    u1 = densities[:, 0]
    u2 = densities[:, 1]
    delta = compute_delta(densities)
    conf = compute_confidence(densities)
    ent = compute_entropy(densities)
    return {
        'std_u1': float(np.std(u1)),
        'std_u2': float(np.std(u2)),
        'mean_confidence': float(np.mean(conf)),
        'median_confidence': float(np.median(conf)),
        'mean_entropy': float(np.mean(ent)),
        'median_entropy': float(np.median(ent)),
        'delta_mean': float(np.mean(delta)),
        'delta_std': float(np.std(delta)),
    }


def extract_zero_isolines(vertices: np.ndarray, faces: np.ndarray, values: np.ndarray) -> List[np.ndarray]:
    """
    Use matplotlib's triangulation contour to extract 0-level polylines.
    Returns list of (M_i, 2) arrays.
    """
    triangulation = tri.Triangulation(vertices[:, 0], vertices[:, 1], faces)
    fig = plt.figure()
    try:
        ax = fig.add_subplot(111)
        cs = ax.tricontour(triangulation, values, levels=[0.0])
        polylines: List[np.ndarray] = []
        for c in cs.collections:
            for path in c.get_paths():
                v = path.vertices  # (M, 2)
                if v.shape[0] >= 2:
                    polylines.append(v.copy())
        # Clean up artists
        for c in cs.collections:
            c.remove()
    finally:
        plt.close(fig)
    return polylines


def polyline_length(poly: np.ndarray) -> float:
    diffs = np.diff(poly, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def polyline_area(poly: np.ndarray) -> float:
    """Shoelace area (works best if closed; for open lines returns projected area)."""
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) 