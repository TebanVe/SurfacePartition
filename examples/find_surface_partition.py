#!/usr/bin/env python3
import os
import sys
import time
import argparse
import yaml
import h5py
import datetime
import logging
import getpass
import platform
import socket
import numpy as np

# Add src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from logging_config import setup_logging, get_logger
from core.pyslsqp_optimizer import PySLSQPOptimizer, RefinementTriggered
from projection_iterative import (
	orthogonal_projection_iterative,
	create_initial_condition_with_projection,
	validate_projection_result,
)
from core.interpolation import nearest_neighbor_interpolate


def optimize_surface_partition(provider, config, solution_dir=None):
	logger = get_logger(__name__)
	initial_n_partitions = config.n_partitions
	timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
	refinement_levels = getattr(config, 'refinement_levels', 1)

	outdir = f"results/run_{timestamp}_npart{initial_n_partitions}_ref{refinement_levels}_lam{getattr(config, 'lambda_penalty', 0.0)}_seed{config.seed}"
	os.makedirs(outdir, exist_ok=True)
	logfile_path = os.path.join(outdir, 'run.log')

	# root logger to file
	root_logger = logging.getLogger()
	file_handler = logging.FileHandler(logfile_path)
	file_handler.setLevel(logging.DEBUG)
	file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
	root_logger.addHandler(file_handler)

	results = []
	logger.info(f"Starting surface partition optimization with {refinement_levels} refinement levels")

	for level in range(refinement_levels):
		logger.info("=" * 80)
		logger.info(f"Refinement Level {level+1}/{refinement_levels}")
		logger.info("=" * 80)

		mesh = provider.build()
		from core.tri_mesh import TriMesh  # type: ignore
		assert isinstance(mesh, TriMesh)
		mesh.compute_matrices()
		stats = mesh.get_mesh_statistics()
		epsilon = np.sqrt(stats['mean_triangle_area']) if stats['mean_triangle_area'] > 0 else 1e-2
		logger.info(f"epsilon set to sqrt(mean_triangle_area) = {epsilon:.3e}")

		total_area = getattr(provider, 'theoretical_total_area', None)
		total_area = provider.theoretical_total_area() if callable(total_area) else float(np.sum(mesh.v))

		optimizer = PySLSQPOptimizer(K=mesh.K, M=mesh.M, v=mesh.v, n_partitions=config.n_partitions,
									epsilon=epsilon, total_area=total_area,
									lambda_penalty=getattr(config, 'lambda_penalty', 0.0),
									refine_patience=int(getattr(config, 'refine_patience', 30)),
									refine_delta_energy=float(getattr(config, 'refine_delta_energy', 1e-4)),
									refine_grad_tol=float(getattr(config, 'refine_grad_tol', 1e-2)),
									refine_constraint_tol=float(getattr(config, 'refine_constraint_tol', 1e-2)),
									logger=logger)

		N = len(mesh.v)
		if level == 0:
			x0 = create_initial_condition_with_projection(N, config.n_partitions, mesh.v, seed=config.seed, method="iterative")
		else:
			prev = results[-1]
			x0 = nearest_neighbor_interpolate(prev['mesh'].vertices, mesh.vertices, prev['x_opt'], config.n_partitions)
			A = x0.reshape(N, config.n_partitions)
			A = orthogonal_projection_iterative(A, np.ones(config.n_partitions), np.sum(mesh.v) / config.n_partitions * np.ones(config.n_partitions), mesh.v, max_iter=100, tol=1e-8)
			x0 = A.flatten()

		start = time.time()
		try:
			x_opt, success = optimizer.optimize(
				x0=x0,
				maxiter=getattr(config, 'max_iter', 1000),
				ftol=float(getattr(config, 'tol', 1e-6)),
				use_analytic=getattr(config, 'use_analytic', True),
				results_dir=outdir,
				run_name=f"pyslsqp_part{config.n_partitions}_level{level}",
				is_mesh_refinement=(level > 0),
				save_itr=getattr(config, 'pyslsqp_save_itr', 'major')
			)
		except RefinementTriggered:
			logger.info(f"Refinement triggered early at level {level+1}")
			x_opt = getattr(optimizer, 'prev_x', x0)
			success = False
		elapsed = time.time() - start

		results.append({
			'level': level,
			'mesh': mesh,
			'epsilon': epsilon,
			'x_opt': x_opt,
			'energy': optimizer.compute_energy(x_opt),
			'iterations': len(optimizer.log.get('iterations', [])),
			'time': elapsed,
			'success': success,
		})

	# Save final solution
	final = results[-1]
	x_opt = final['x_opt']
	mesh = final['mesh']
	solution_path = os.path.join(solution_dir or outdir, f"surface_part{config.n_partitions}_{timestamp}.h5")
	with h5py.File(solution_path, 'w') as f:
		f.create_dataset('x_opt', data=x_opt)
		f.create_dataset('x0', data=x0)
		f.create_dataset('vertices', data=mesh.vertices)
		f.create_dataset('faces', data=mesh.faces, dtype='i4')
		f.attrs['n_partitions'] = config.n_partitions

	# Save metadata
	meta = {
		'input_parameters': {
			'refinement_levels': int(refinement_levels),
			'use_analytic': bool(getattr(config, 'use_analytic', True)),
			'lambda_penalty': float(getattr(config, 'lambda_penalty', 0.0)),
			'seed': int(config.seed),
		},
		'final_mesh_stats': mesh.get_mesh_statistics(),
		'final_epsilon': float(final['epsilon']),
		'final_energy': float(final['energy']),
		'final_iterations': int(final['iterations']),
		'run_time_seconds': float(final['time']),
		'success': bool(final['success']),
		'datetime': timestamp,
		'user': getpass.getuser(),
		'hostname': socket.gethostname(),
		'platform': platform.platform(),
		'python_version': platform.python_version(),
		'solution_path': solution_path,
		'optimizer': 'PySLSQP'
	}
	with open(os.path.join(outdir, 'metadata.yaml'), 'w') as f:
		yaml.dump(meta, f)
	print(f"Surface partition optimization complete. See {logfile_path} for details.\n")
	print(f"Results saved in: {outdir}")
	return results


def main():
	from config import Config
	from surfaces.ring import RingMeshProvider
	parser = argparse.ArgumentParser(description='Generic surface partition optimization')
	parser.add_argument('--input', type=str, help='Path to input YAML')
	parser.add_argument('--solution-dir', type=str, help='Directory to save solutions')
	parser.add_argument('--surface', type=str, default='ring', help='Surface type (ring for now)')
	args = parser.parse_args()

	setup_logging(log_level='INFO', log_to_console=True, log_to_file=False)
	logger = get_logger(__name__)

	if args.input:
		with open(args.input, 'r') as f:
			params = yaml.safe_load(f)
		config = Config(params)
	else:
		config = Config()

	if args.surface == 'ring':
		provider = RingMeshProvider(config.n_radial, config.n_angular, config.r_inner, config.r_outer)
	else:
		raise ValueError(f"Unsupported surface type: {args.surface}")

	optimize_surface_partition(provider, config, solution_dir=args.solution_dir)


if __name__ == '__main__':
	main() 