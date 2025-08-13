import numpy as np
from typing import Tuple, Optional

try:
	from ..logging_config import get_logger, log_performance
except Exception:
	import sys, os
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
	from logging_config import get_logger, log_performance

# PySLSQP import
try:
	import pyslsqp
	PYSLSQP_AVAILABLE = True
except Exception:
	PYSLSQP_AVAILABLE = False


class RefinementTriggered(Exception):
	pass


class SurfaceAgnosticOptimizer:
	"""
	PySLSQP-based optimizer for partition problems on triangle meshes.
	Surface-agnostic: requires only K, M, v, total_area, epsilon, n_partitions.
	"""
	def __init__(self, K: np.ndarray, M: np.ndarray, v: np.ndarray, n_partitions: int,
				epsilon: float, total_area: Optional[float] = None,
				lambda_penalty: float = 0.0,
				refine_patience: int = 30, refine_delta_energy: float = 1e-4,
				refine_grad_tol: float = 1e-2, refine_constraint_tol: float = 1e-2,
				logger=None):
		if not PYSLSQP_AVAILABLE:
			raise ImportError("PySLSQP is not available. Please install it first.")
		self.logger = logger or get_logger(__name__)
		self.K = K
		self.M = M
		self.v = v
		self.n_partitions = n_partitions
		self.epsilon = epsilon
		self.lambda_penalty = lambda_penalty
		self.total_area = float(total_area) if total_area is not None else float(np.sum(v))
		self.target_area = self.total_area / n_partitions
		self.refine_patience = refine_patience
		self.refine_delta_energy = refine_delta_energy
		self.refine_grad_tol = refine_grad_tol
		self.refine_constraint_tol = refine_constraint_tol
		self.log = {'iterations': [], 'area_evolution': [], 'energy_changes': []}

	def compute_energy(self, x: np.ndarray) -> float:
		N = len(self.v)
		n = self.n_partitions
		phi = x.reshape(N, n)
		total = 0.0
		for i in range(n):
			u = phi[:, i]
			grad_term = self.epsilon * float(u.T @ (self.K @ u))
			interface_vec = u ** 2 * (1 - u) ** 2
			interface_term = (1 / self.epsilon) * float(interface_vec.T @ (self.M @ interface_vec))
			total += grad_term + interface_term
		if self.lambda_penalty > 0:
			for i in range(n):
				u = phi[:, i]
				mu = float(np.mean(u))
				var = float(np.var(u))
				total += self.lambda_penalty * (1.0 - var / (mu * (1 - mu) + 1e-8))
		return total

	def compute_gradient(self, x: np.ndarray) -> np.ndarray:
		N = len(self.v)
		n = self.n_partitions
		phi = x.reshape(N, n)
		grad = np.zeros_like(x)
		G = grad.reshape(N, n)
		for i in range(n):
			u = phi[:, i]
			grad_grad = 2 * self.epsilon * (self.K @ u)
			interface_vec = u ** 2 * (1 - u) ** 2
			grad_interface = (2 / self.epsilon) * (self.M @ interface_vec) * (1 - 2 * u)
			G[:, i] = grad_grad + grad_interface
		if self.lambda_penalty > 0:
			for i in range(n):
				u = phi[:, i]
				mu = float(np.mean(u))
				G[:, i] += self.lambda_penalty * (-2 * (u - mu) / N) / (mu * (1 - mu) + 1e-8)
		return grad

	def constraint_fun(self, x: np.ndarray) -> np.ndarray:
		N = len(self.v)
		n = self.n_partitions
		phi = x.reshape(N, n)
		row_sums = np.sum(phi, axis=1)[:-1] - 1.0
		area_sums = self.v @ phi
		area_constraints = area_sums[:-1] - self.target_area
		return np.concatenate([row_sums, area_constraints])

	def constraint_jac(self, x: np.ndarray) -> np.ndarray:
		N = len(self.v)
		n = self.n_partitions
		row_sum_jac = np.zeros((N - 1, N * n))
		for i in range(N - 1):
			row_sum_jac[i, i::N] = 1.0
		area_jac = np.zeros((n - 1, N * n))
		for i in range(n - 1):
			area_jac[i, i * N:(i + 1) * N] = self.v
		return np.vstack([row_sum_jac, area_jac])

	def compute_area_evolution(self, x: np.ndarray) -> np.ndarray:
		N = len(self.v)
		n = self.n_partitions
		phi = x.reshape(N, n)
		return self.v @ phi

	@log_performance("optimization")
	def optimize(self, x0: np.ndarray, maxiter: int = 1000, ftol: float = 1e-6,
				eps: float = 1e-8, disp: bool = False, use_analytic: bool = True,
				log_frequency: int = 50, use_last_valid_iterate: bool = True,
				is_mesh_refinement: bool = False, results_dir: Optional[str] = None,
				run_name: Optional[str] = None, save_itr: Optional[str] = None) -> Tuple[np.ndarray, bool]:
		# Minimal wrapper around pyslsqp
		N = len(self.v)
		if use_analytic:
			grad = self.compute_gradient
		else:
			grad = None
		problem_size = len(x0)
		xl = np.zeros(problem_size)
		xu = np.ones(problem_size)
		meq = N - 1 + self.n_partitions - 1
		params = {
			'obj': self.compute_energy,
			'grad': grad,
			'con': self.constraint_fun,
			'jac': self.constraint_jac,
			'meq': meq,
			'xl': xl,
			'xu': xu,
			'maxiter': maxiter,
			'acc': ftol,
			'iprint': 0 if not disp else 2,
			'summary_filename': None,
			'save_vars': ['x', 'objective', 'constraints', 'gradient', 'jacobian'],
			'save_itr': save_itr or 'major',
			'save_filename': None
		}
		res = pyslsqp.optimize(x0, **params)
		x_opt = res['x']
		return x_opt, bool(res['success']) 