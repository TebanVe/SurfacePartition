import os
import datetime
import time
import logging
from typing import Optional, Tuple, List

import h5py
import numpy as np

try:
	from ..logging_config import get_logger
	from .pyslsqp_optimizer import RefinementTriggered
	except_import_ok = True
except Exception:
	# Fallback when executed standalone
	except_import_ok = False
	import sys
	import os as _os
	sys.path.append(_os.path.join(_os.path.dirname(__file__), '..'))
	from logging_config import get_logger  # type: ignore
	from core.pyslsqp_optimizer import RefinementTriggered  # type: ignore

try:
	from ..projection_iterative import orthogonal_projection_iterative
except Exception:
	from projection_iterative import orthogonal_projection_iterative


class ProjectedGradientOptimizer:
	"""
	Projected Gradient Descent optimizer with per-step projection onto
	partition unity and equal-area constraints. Produces analyzer-compatible
	summary and internal-data artifacts.
	"""

	def __init__(
		self,
		K: np.ndarray,
		M: np.ndarray,
		v: np.ndarray,
		n_partitions: int,
		epsilon: float,
		total_area: Optional[float] = None,
		lambda_penalty: float = 0.0,
		refine_patience: int = 30,
		refine_delta_energy: float = 1e-4,
		refine_grad_tol: float = 1e-2,
		refine_constraint_tol: float = 1e-2,
		logger=None,
	):
		self.logger = logger or get_logger(__name__)
		self.K = K
		self.M = M
		self.v = v
		self.n_partitions = int(n_partitions)
		self.epsilon = float(epsilon)
		self.lambda_penalty = float(lambda_penalty)
		# Prefer geometric total_area from v; fall back to provided
		self.total_area = float(total_area) if total_area is not None else float(np.sum(v))
		self.target_area = self.total_area / self.n_partitions
		# Precompute total weight and penalty defaults
		self.W = float(np.sum(self.v))
		self.mu_target = 1.0 / self.n_partitions
		self.penalty_target_mode = 'fixed'  # or 'adaptive'
		self.penalty_eps = 1e-8
		
		# Refinement criteria
		self.refine_patience = int(refine_patience)
		self.refine_delta_energy = float(refine_delta_energy)
		self.refine_grad_tol = float(refine_grad_tol)
		self.refine_constraint_tol = float(refine_constraint_tol)
		
		# Logging cache
		self.log = {
			'iterations': [],
			'energy_changes': [],
			'area_evolution': [],
			'gnorm': [],
			'feas': [],
		}
		self.prev_x = None
		self.curr_x = None

	def compute_energy(self, x: np.ndarray) -> float:
		N = len(self.v)
		n = self.n_partitions
		phi = x.reshape(N, n)
		total_energy = 0.0
		for i in range(n):
			u = phi[:, i]
			grad_term = self.epsilon * float(u.T @ (self.K @ u))
			interface_vec = u ** 2 * (1 - u) ** 2
			# Keep consistency with PySLSQP energy structure
			interface_term = (1 / self.epsilon) * float(interface_vec.T @ (self.M @ interface_vec))
			total_energy += grad_term + interface_term
		if self.lambda_penalty > 0:
			for i in range(n):
				u = phi[:, i]
				# Weighted mean and variance
				mu = float((self.v @ u) / self.W)
				center = u - mu
				var_w = float(((center * self.v) @ center) / self.W)
				# Target variance (fixed or adaptive)
				if self.penalty_target_mode == 'adaptive':
					T = mu * (1.0 - mu)
				else:
					mu_t = self.mu_target
					T = mu_t * (1.0 - mu_t)
				T_eff = T + self.penalty_eps
				total_energy += self.lambda_penalty * (1.0 - var_w / T_eff)
		return total_energy

	def compute_gradient(self, x: np.ndarray) -> np.ndarray:
		N = len(self.v)
		n = self.n_partitions
		phi = x.reshape(N, n)
		g = np.zeros_like(x)
		G = g.reshape(N, n)
		for i in range(n):
			u = phi[:, i]
			grad_grad = 2 * self.epsilon * (self.K @ u)
			interface_vec = u ** 2 * (1 - u) ** 2
			grad_interface = (2 / self.epsilon) * (self.M @ interface_vec) * (1 - 2 * u)
			G[:, i] = grad_grad + grad_interface
		if self.lambda_penalty > 0:
			for i in range(n):
				u = phi[:, i]
				# Weighted statistics
				mu = float((self.v @ u) / self.W)
				center = u - mu
				var_w = float(((center * self.v) @ center) / self.W)
				# Target variance (fixed or adaptive)
				if self.penalty_target_mode == 'adaptive':
					T = mu * (1.0 - mu)
				else:
					mu_t = self.mu_target
					T = mu_t * (1.0 - mu_t)
				T_eff = T + self.penalty_eps
				# Gradient of weighted variance: (2/W) diag(v) (u - mu*1)
				grad_var = (2.0 / self.W) * (self.v * center)
				if self.penalty_target_mode == 'adaptive':
					# Full adaptive gradient: -lambda [ (1/T) grad_var - (Var/T^2) (1-2mu) (v/W) ]
					term1 = grad_var / T_eff
					term2 = (var_w / (T_eff * T_eff)) * (1.0 - 2.0 * mu) * (self.v / self.W)
					G[:, i] += -self.lambda_penalty * (term1 - term2)
				else:
					# Fixed target gradient: -lambda * (1/T) * grad_var
					G[:, i] += -self.lambda_penalty * (grad_var / T_eff)
		return g

	def constraint_fun(self, x: np.ndarray) -> np.ndarray:
		N = len(self.v)
		n = self.n_partitions
		phi = x.reshape(N, n)
		row_sums = np.sum(phi, axis=1)[:-1] - 1.0
		area_sums = self.v @ phi
		area_constraints = area_sums[:-1] - self.target_area
		return np.concatenate([row_sums, area_constraints])

	def _save_iteration_h5(self, h5, k: int, x: np.ndarray, g: np.ndarray, f: float, cvec: np.ndarray, save_vars: List[str]):
		grp = h5.create_group(f'iter_{k}')
		if 'x' in save_vars:
			grp.create_dataset('x', data=x)
		if 'gradient' in save_vars:
			grp.create_dataset('gradient', data=g)
		if 'objective' in save_vars:
			grp.create_dataset('objective', data=f)
		if 'constraints' in save_vars:
			grp.create_dataset('constraints', data=cvec)
		grp.create_dataset('ismajor', data=True)

	def _append_summary_line(self, fh, k: int, f: float, gnorm: float, cnorm: float, feas: float, step: float):
		# Columns (9 tokens): MAJOR-idx, NFEV, NGEV, OBJFUN, GNORM, CNORM, FEAS, OPT, STEP (OPT dummy 0)
		line = f"{k} 0 0 {f:.16e} {gnorm:.16e} {cnorm:.16e} {feas:.16e} 0 {step:.16e}\n"
		fh.write(line)

	def optimize(
		self,
		x0: Optional[np.ndarray] = None,
		maxiter: int = 1000,
		step0: float = 1.0,
		armijo_c: float = 1e-4,
		backtrack_rho: float = 0.5,
		projection_max_iter: int = 100,
		projection_tol: float = 1e-8,
		log_frequency: int = 50,
		results_dir: Optional[str] = None,
		run_name: Optional[str] = None,
		is_mesh_refinement: bool = False,
		data_save_stride: int = 1,
		data_save_vars: Optional[List[str]] = None,
		save_first_last: bool = True,
		refine_trigger_mode: str = 'full',
		refine_gnorm_patience: int = 30,
		refine_gnorm_delta: float = 1e-4,
		refine_feas_patience: int = 30,
		refine_feas_delta: float = 1e-6,
		enable_refinement_triggers: bool = True,
	) -> Tuple[np.ndarray, bool]:
		"""
		Run PGD with per-step projection and Armijo backtracking.
		"""
		N = len(self.v)
		n = self.n_partitions
		if x0 is None:
			# Random simplex init then project
			x0 = np.random.rand(N * n)
			A0 = x0.reshape(N, n)
			c = np.ones(n)
			d = (np.sum(self.v) / n) * np.ones(n)
			A0 = orthogonal_projection_iterative(A0, c, d, self.v, max_iter=projection_max_iter, tol=projection_tol, logger=self.logger)
			x0 = A0.flatten()

		if results_dir is None:
			results_dir = "results"
		if run_name is None:
			run_name = f"pgd_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
		os.makedirs(results_dir, exist_ok=True)
		summary_filename = os.path.join(results_dir, f"{run_name}_summary.out")
		internal_data_filename = os.path.join(results_dir, f"{run_name}_internal_data.hdf5")
		self.optimization_run_name = run_name
		self.optimization_results_dir = results_dir
		self.summary_file = summary_filename
		self.internal_data_file = internal_data_filename

		# Build a quiet logger for projection calls to avoid per-step spam
		proj_logger = get_logger(__name__ + ".projection")
		proj_logger.setLevel(logging.WARNING)

		A = x0.reshape(N, n).copy()
		A = np.clip(A, 1e-8, 1 - 1e-8)
		c = np.ones(n)
		d = (np.sum(self.v) / n) * np.ones(n)
		A = orthogonal_projection_iterative(A, c, d, self.v, max_iter=projection_max_iter, tol=projection_tol, logger=proj_logger)
		x = A.flatten()

		E = self.compute_energy(x)
		best_x = x.copy()
		best_E = E
		self.prev_x = None
		self.curr_x = x.copy()

		self.logger.info("Starting PGD optimization")
		start_time = time.time()

		# Determine what to save in HDF5
		save_vars_h5 = ['x'] if data_save_vars is None else list(data_save_vars)
		stride = max(1, int(data_save_stride))

		# Open files
		with open(summary_filename, 'w') as summary_fh, h5py.File(internal_data_filename, 'w') as h5f:
			# Optional header line (analyzer ignores lines starting with 'MAJOR')
			summary_fh.write("MAJOR NFEV NGEV OBJFUN GNORM CNORM FEAS OPT STEP\n")

			for k in range(maxiter):
				# Gradient at current x
				g = self.compute_gradient(x)
				# Backtracking line search
				step = float(step0)
				accepted = False
				while True:
					A_trial = x.reshape(N, n) - step * g.reshape(N, n)
					A_trial = np.clip(A_trial, 1e-8, 1 - 1e-8)
					A_trial = orthogonal_projection_iterative(
						A_trial, c, d, self.v, max_iter=projection_max_iter, tol=projection_tol, logger=proj_logger
					)
					x_trial = A_trial.flatten()
					E_trial = self.compute_energy(x_trial)
					# Armijo condition with ||g||^2 surrogate
					if E_trial <= E - armijo_c * step * float(np.dot(g, g)):
						accepted = True
						x = x_trial
						E = E_trial
						break
					step *= backtrack_rho
					if step < 1e-12:
						# Unable to make progress
						break

				# Recompute gradient and constraints at the accepted iterate (or current if not accepted)
				g_post = self.compute_gradient(x)
				cvec_post = self.constraint_fun(x)
				gnorm_post = float(np.linalg.norm(g_post))
				cnorm_post = float(np.linalg.norm(cvec_post))
				feas_post = float(np.max(np.abs(cvec_post))) if cvec_post.size > 0 else 0.0

				# Save iteration (post-accept values) according to stride/vars
				should_save_iter = (k % stride == 0) or (save_first_last and (k == 0 or k == maxiter - 1))
				if should_save_iter:
					self._save_iteration_h5(h5f, k, x, g_post, E, cvec_post, save_vars_h5)
				self._append_summary_line(summary_fh, k, E, gnorm_post, cnorm_post, feas_post, step)
				summary_fh.flush()
				h5f.flush()

				# Track logs
				self.log['iterations'].append(k)
				areas = self.v @ x.reshape(N, n)
				self.log['area_evolution'].append(areas.copy())
				self.log['energy_changes'].append(0.0 if k == 0 else (E - best_E))
				self.log['gnorm'].append(gnorm_post)
				self.log['feas'].append(feas_post)
				self.prev_x = self.curr_x
				self.curr_x = x.copy()

				# Best-so-far
				if E < best_E:
					best_E = E
					best_x = x.copy()

				# Progress log
				if k % max(1, log_frequency) == 0:
					self.logger.info(f"  Iteration {k}: Energy={E:.12e}")
					self.logger.info(f"    GNORM={gnorm_post:.6e}, FEAS={feas_post:.6e}, STEP={step:.3e}")
					areas_log = self.v @ x.reshape(N, n)
					self.logger.info(f"    Target area per partition: {self.target_area:.6e}")
					self.logger.info(f"    Current partition areas: {areas_log}")

				# Refinement trigger check
				if enable_refinement_triggers and (k + 1 >= self.refine_patience):
					recent = self.log['energy_changes'][-self.refine_patience:]
					stable = all(abs(de) < self.refine_delta_energy for de in recent)
					if stable:
						if refine_trigger_mode == 'energy':
							self.logger.info(f"Refinement triggered at iteration {k} (energy criterion)")
							raise RefinementTriggered()
						else:
							# plateau checks for gnorm and feas
							gn_ok = (gnorm_post < self.refine_grad_tol)
							fe_ok = (feas_post < self.refine_constraint_tol)
							if not gn_ok and len(self.log['gnorm']) >= refine_gnorm_patience:
								recent_g = self.log['gnorm'][-refine_gnorm_patience:]
								gn_plateau = all(abs(recent_g[i] - recent_g[i-1]) < refine_gnorm_delta for i in range(1, len(recent_g)))
								gn_ok = gn_ok or gn_plateau
							if not fe_ok and len(self.log['feas']) >= refine_feas_patience:
								recent_f = self.log['feas'][-refine_feas_patience:]
								fe_plateau = all(abs(recent_f[i] - recent_f[i-1]) < refine_feas_delta for i in range(1, len(recent_f)))
								fe_ok = fe_ok or fe_plateau
							if gn_ok and fe_ok:
								self.logger.info(f"Refinement triggered at iteration {k}")
								raise RefinementTriggered()

		# Final summary log
		elapsed = time.time() - start_time
		self.logger.info(f"PGD optimization completed: Success=True")
		self.logger.info(f"  Summary saved to: {summary_filename}")
		self.logger.info(f"  Internal data saved to: {internal_data_filename}")
		self.logger.info(f"  optimization completed: {elapsed:.3f}s")

		# Return best found
		return best_x.copy(), True 