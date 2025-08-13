#!/usr/bin/env python3
"""
Main program for finding optimal partition of a ring using Γ-convergence.
"""

import os
# Control thread usage for numerical libraries to prevent oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import sys
import time
import argparse
import h5py
import yaml
import datetime
import getpass
import platform
import socket
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import Config
from ring_mesh import RingMesh
from pyslsqp_optimizer import PySLSQPOptimizer, RefinementTriggered
from projection_iterative import (
    orthogonal_projection_iterative,
    create_initial_condition_with_projection,
    validate_projection_result
)
from logging_config import setup_logging, get_logger

def load_initial_condition(h5_path: str, mesh: RingMesh, n_partitions: int, logger=None) -> tuple:
    """
    Load initial condition from HDF5 file and validate it.
    
    Args:
        h5_path: Path to HDF5 file containing initial condition
        mesh: RingMesh object
        n_partitions: Number of partitions
        logger: Logger instance
        
    Returns:
        Tuple of (x0, hot_start_data)
    """
    if logger is None:
        logger = get_logger(__name__)
    
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Initial condition file not found: {h5_path}")
    
    try:
        with h5py.File(h5_path, 'r') as f:
            x0 = f['x_opt'][:]
            logger.info(f"Loaded initial condition with shape: {x0.shape}")
            
            # Validate dimensions
            N = len(mesh.v)
            expected_shape = N * n_partitions
            if x0.shape[0] != expected_shape:
                raise ValueError(f"Initial condition shape {x0.shape[0]} doesn't match expected {expected_shape}")
            
            # Validate constraints
            if not validate_projection_result(x0.reshape(N, n_partitions), mesh.v, 
                                           mesh.v.sum() / n_partitions * np.ones(n_partitions), logger=logger):
                logger.warning("Initial condition doesn't satisfy constraints, will project it")
                x0 = create_initial_condition_with_projection(N, n_partitions, mesh.v, method="iterative")
            
            # Check for hot-start data
            hot_start_data = {'capable': False, 'file': None}
            if 'hot_start_file' in f.attrs:
                hot_start_file = f.attrs['hot_start_file']
                if os.path.exists(hot_start_file):
                    hot_start_data = {'capable': True, 'file': hot_start_file}
                    logger.info(f"Found hot-start file: {hot_start_file}")
            
            return x0, hot_start_data
            
    except Exception as e:
        logger.error(f"Failed to load initial condition: {e}")
        raise

def validate_initial_condition(x0: np.ndarray, v: np.ndarray, n_partitions: int, logger=None) -> bool:
    """
    Validate that initial condition satisfies constraints.
    
    Args:
        x0: Initial condition vector
        v: Mass matrix column sums
        n_partitions: Number of partitions
        logger: Logger instance
        
    Returns:
        True if valid, False otherwise
    """
    if logger is None:
        logger = get_logger(__name__)
    
    N = len(v)
    expected_shape = N * n_partitions
    
    if x0.shape[0] != expected_shape:
        logger.error(f"Initial condition shape {x0.shape[0]} doesn't match expected {expected_shape}")
        return False
    
    # Reshape and validate constraints
    A = x0.reshape(N, n_partitions)
    target_areas = v.sum() / n_partitions * np.ones(n_partitions)
    
    valid = validate_projection_result(A, v, target_areas, logger=logger)
    
    if valid:
        logger.info("Initial condition validation passed")
    else:
        logger.warning("Initial condition validation failed")
    
    return valid

def check_analytic_vs_fd_gradient(optimizer, x0, logger=None, eps=1e-6, n_check=10):
    """
    Check analytic vs finite difference gradients.
    
    Args:
        optimizer: PySLSQPOptimizer instance
        x0: Initial condition
        logger: Logger instance
        eps: Finite difference step size
        n_check: Number of random points to check
    """
    if logger is None:
        logger = get_logger(__name__)
    
    logger.info("Checking analytic vs finite-difference gradients...")
    
    def finite_difference_gradient(f, x, eps=1e-6):
        """Compute finite difference gradient."""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
        return grad
    
    # Check at random points
    errors = []
    for i in range(n_check):
        # Perturb x0 slightly
        x_test = x0 + 0.01 * np.random.randn(len(x0))
        
        # Compute gradients
        grad_analytic = optimizer.compute_gradient(x_test)
        grad_fd = finite_difference_gradient(optimizer.compute_energy, x_test, eps)
        
        # Compute relative error
        rel_error = np.linalg.norm(grad_analytic - grad_fd) / (np.linalg.norm(grad_analytic) + 1e-10)
        errors.append(rel_error)
        
        if rel_error > 1e-3:
            logger.warning(f"Large gradient error at point {i}: {rel_error:.2e}")
    
    avg_error = np.mean(errors)
    max_error = np.max(errors)
    logger.info(f"Gradient check completed: avg_error={avg_error:.2e}, max_error={max_error:.2e}")
    
    if avg_error > 1e-4:
        logger.warning("Gradient errors are larger than expected")

def interpolate_solution(old_x: np.ndarray, old_mesh: RingMesh, new_mesh: RingMesh, n_partitions: int, logger=None) -> np.ndarray:
    """
    Interpolate a solution defined on an old mesh to a new mesh using nearest-neighbor mapping.
    
    Args:
        old_x: Flattened solution vector on the old mesh (size N_old * n_partitions)
        old_mesh: Previous `RingMesh`
        new_mesh: Current `RingMesh`
        n_partitions: Number of partitions
        logger: Optional logger
    Returns:
        Flattened solution vector on the new mesh (size N_new * n_partitions)
    """
    if logger is None:
        logger = get_logger(__name__)
    old_vertices = old_mesh.vertices
    new_vertices = new_mesh.vertices
    N_old = old_vertices.shape[0]
    N_new = new_vertices.shape[0]
    if old_x.shape[0] != N_old * n_partitions:
        raise ValueError(f"old_x has size {old_x.shape[0]} but expected {N_old * n_partitions}")
    old_phi = old_x.reshape(N_old, n_partitions)
    new_phi = np.zeros((N_new, n_partitions))
    for i in range(N_new):
        new_point = new_vertices[i]
        # nearest neighbor in Euclidean coordinates
        distances = np.linalg.norm(old_vertices - new_point, axis=1)
        closest_idx = int(np.argmin(distances))
        new_phi[i, :] = old_phi[closest_idx, :]
    return new_phi.flatten()

def optimize_partition_ring(config, solution_dir=None):
    """
    Optimize partition on a ring mesh using PySLSQP optimizer.
    
    Args:
        config: Configuration object containing mesh and optimization parameters
        solution_dir: Optional directory to save solution files
    """
    initial_n_radial = config.n_radial
    initial_n_angular = config.n_angular
    initial_n_partitions = config.n_partitions
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    use_analytic = getattr(config, 'use_analytic', True)
    refinement_levels = getattr(config, 'refinement_levels', 1)
    
    # Build mesh info strings for naming
    if refinement_levels > 1:
        final_n_radial = initial_n_radial + (refinement_levels - 1) * getattr(config, 'n_radial_increment', 0)
        final_n_angular = initial_n_angular + (refinement_levels - 1) * getattr(config, 'n_angular_increment', 0)
        n_radial_info = f"{initial_n_radial}-{final_n_radial}_incr{getattr(config, 'n_radial_increment', 0)}"
        n_angular_info = f"{initial_n_angular}-{final_n_angular}_inca{getattr(config, 'n_angular_increment', 0)}"
    else:
        n_radial_info = f"{initial_n_radial}"
        n_angular_info = f"{initial_n_angular}"
    
    outdir = f'results/run_{timestamp}_npart{initial_n_partitions}_nr{n_radial_info}_na{n_angular_info}_lam{getattr(config, "lambda_penalty", 1.0)}_seed{config.seed}'
    os.makedirs(outdir, exist_ok=True)
    logfile_path = os.path.join(outdir, 'run.log')
    logger = get_logger(__name__)
    
    # Add file handler to the root logger so all components log to the file
    root_logger = logging.getLogger()
    file_handler = logging.FileHandler(logfile_path)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    if solution_dir:
        os.makedirs(solution_dir, exist_ok=True)
        logger.info(f"Solutions will be stored in: {solution_dir}")
    
    results = []
    logger.info(f"Starting ring partition optimization with {refinement_levels} refinement levels")
    logger.info(f"Results will be saved in: {outdir}")
    
    for level in range(refinement_levels):
        logger.info(f"{'='*80}")
        logger.info(f"Refinement Level {level + 1}/{refinement_levels}")
        logger.info(f"{'='*80}")
        
        # Create mesh
        mesh_params = config.get_mesh_parameters()
        logger.info(f"Creating mesh with parameters: {mesh_params}")
        
        mesh = RingMesh(**mesh_params)
        logger.info(f"Mesh created: {mesh.get_vertex_count()} vertices, {mesh.get_triangle_count()} triangles")
        
        # Compute matrices
        logger.info("Computing FEM matrices...")
        mesh.compute_matrices()
        logger.info("Matrices computed successfully")
        
        # Get mesh statistics and set epsilon
        mesh_stats = mesh.get_mesh_statistics()
        # Use square root of mean triangle area as epsilon (characteristic length scale)
        epsilon = np.sqrt(mesh_stats['mean_triangle_area'])
        logger.info(f"Setting epsilon to sqrt(mean_triangle_area): {epsilon:.6e}")
        
        # Create optimizer
        optimizer = PySLSQPOptimizer(
            K=mesh.K,
            M=mesh.M,
            v=mesh.v,
            n_partitions=config.n_partitions,
            epsilon=epsilon,
            r_inner=config.r_inner,
            r_outer=config.r_outer
        )
        logger.info("PySLSQP optimizer created")
        
        N = len(mesh.v)
        
        # Initialize solution
        if level == 0:
            if getattr(config, 'use_custom_initial_condition', False) and getattr(config, 'initial_condition_path', None):
                try:
                    x0, hot_start_data = load_initial_condition(
                        config.initial_condition_path, mesh, config.n_partitions, logger
                    )
                    logger.info("Successfully loaded initial condition")
                    hot_start_file = hot_start_data.get('file') if hot_start_data.get('capable', False) else None
                except Exception as e:
                    logger.error(f"Failed to load initial condition: {e}")
                    logger.info("Falling back to random initialization")
                    x0 = create_initial_condition_with_projection(
                        N, config.n_partitions, mesh.v, seed=config.seed, method="iterative"
                    )
                    hot_start_file = None
            else:
                logger.info("Creating random initial condition using orthogonal projection")
                x0 = create_initial_condition_with_projection(
                    N, config.n_partitions, mesh.v, seed=config.seed, method="iterative"
                )
                hot_start_file = None
        else:
            # For refinement levels, interpolate from previous solution if mesh changed
            prev_mesh = results[-1]['mesh']
            if (config.n_radial_increment == 0 and getattr(config, 'n_angular_increment', 0) == 0):
                logger.info("Mesh resolution unchanged. Reusing previous optimized solution as initial guess.")
                x0 = results[-1]['x_opt'].copy()
            else:
                logger.info("Mesh resolution changed. Interpolating solution from previous mesh to current mesh...")
                x0 = interpolate_solution(
                    old_x=results[-1]['x_opt'],
                    old_mesh=prev_mesh,
                    new_mesh=mesh,
                    n_partitions=config.n_partitions,
                    logger=logger
                )
                # Project interpolated solution to feasible region on the new mesh
                logger.info("Projecting interpolated solution to feasible region (orthogonal projection)")
                A = x0.reshape(N, config.n_partitions)
                c = np.ones(config.n_partitions)
                d = mesh.v.sum() / config.n_partitions * np.ones(config.n_partitions)
                A_projected = orthogonal_projection_iterative(A, c, d, mesh.v, max_iter=100, tol=1e-8)
                x0 = A_projected.flatten()
                validate_initial_condition(x0, mesh.v, config.n_partitions, logger)
            hot_start_file = None
        
        # Validate initial condition
        if not validate_initial_condition(x0, mesh.v, config.n_partitions, logger):
            logger.warning("Initial condition validation failed, projecting to feasible region")
            A = x0.reshape(N, config.n_partitions)
            c = np.ones(config.n_partitions)
            d = mesh.v.sum() / config.n_partitions * np.ones(config.n_partitions)
            A_projected = orthogonal_projection_iterative(A, c, d, mesh.v, max_iter=100, tol=1e-8)
            x0 = A_projected.flatten()
        
        # Gradient check (only if using analytic gradients)
        if use_analytic:
            logger.info("Checking analytic vs finite-difference gradients...")
            check_analytic_vs_fd_gradient(optimizer, x0, logger)
        
        # Run optimization
        start_time = time.time()
        try:
            run_name = f"pyslsqp_part{config.n_partitions}_nt{config.n_radial}_np{config.n_angular}_lam{getattr(config, 'lambda_penalty', 0.0)}_seed{config.seed}_level{level}"
            
            is_mesh_refinement = (level > 0) and (
                getattr(config, 'n_radial_increment', 0) > 0 or getattr(config, 'n_angular_increment', 0) > 0
            )
            x_opt, success = optimizer.optimize(
                x0,
                maxiter=getattr(config, 'max_iter', 10),
                ftol=float(getattr(config, 'tol', 1e-6)),
                use_analytic=use_analytic,
                results_dir=outdir,
                run_name=run_name,
                is_mesh_refinement=is_mesh_refinement,
                save_itr=getattr(config, 'pyslsqp_save_itr', 'major')
            )
        except RefinementTriggered:
            logger.info(f"Refinement triggered early at level {level+1}")
            x_opt = getattr(optimizer, 'prev_x', x0)
            success = False
        
        opt_time = time.time() - start_time
        
        # Store results
        results.append({
            'level': level,
            'mesh_params': mesh_params,
            'mesh_stats': mesh_stats,
            'epsilon': epsilon,
            'x_opt': x_opt,
            'energy': optimizer.compute_energy(x_opt),
            'iterations': len(optimizer.log.get('iterations', [])),
            'time': opt_time,
            'success': success,
            'mesh': mesh,
            'optimizer': optimizer,
            'v': mesh.v.copy(),
            'n_vertices': len(mesh.v)
        })
        
        logger.info(f"Results for level {level + 1}:")
        logger.info(f"  Energy: {results[-1]['energy']:.6e}")
        logger.info(f"  Iterations: {results[-1]['iterations']}")
        logger.info(f"  Time: {opt_time:.2f}s")
        logger.info(f"  Success: {success}")
        
        # Refine mesh for next level if needed
        if level < refinement_levels - 1:
            config.n_radial += getattr(config, 'n_radial_increment', 0)
            config.n_angular += getattr(config, 'n_angular_increment', 0)
    
    # Print summary
    logger.info("Refinement Summary:")
    logger.info("=" * 80)
    logger.info(f"{'Level':>6} {'Mesh Size':>12} {'Energy':>12} {'Iterations':>10} {'Time (s)':>10}")
    logger.info("-" * 80)
    for r in results:
        mesh_size = f"{r['mesh_params']['n_radial']}x{r['mesh_params']['n_angular']}"
        logger.info(f"{r['level']+1:6d} {mesh_size:>12} {r['energy']:12.6e} {r['iterations']:10d} {r['time']:10.2f}")
    
    # Save final solution
    final_result = results[-1]
    x_opt = final_result['x_opt']
    mesh = final_result['mesh']
    
    solution_filename = f"ring_part{config.n_partitions}_nr{n_radial_info}_na{n_angular_info}_lam{getattr(config, 'lambda_penalty', 1.0)}_seed{config.seed}_{timestamp}.h5"
    solution_path = os.path.join(solution_dir if solution_dir else outdir, solution_filename)
    
    with h5py.File(solution_path, 'w') as f:
        f.create_dataset('x_opt', data=x_opt)
        f.create_dataset('x0', data=x0)
        f.create_dataset('vertices', data=mesh.vertices)
        f.create_dataset('faces', data=mesh.faces, dtype='i4')
        f.attrs['r_inner'] = config.r_inner
        f.attrs['r_outer'] = config.r_outer
        f.attrs['n_partitions'] = config.n_partitions
    
    logger.info(f"✅ Solution file saved: {solution_path}")
    
    # Save metadata
    meta = {
        'input_parameters': {
            'RING_PARAMS': {
                'n_radial': config.n_radial,
                'n_angular': config.n_angular,
                'r_inner': config.r_inner,
                'r_outer': config.r_outer
            },
            'refinement_levels': refinement_levels,
            'n_radial_increment': getattr(config, 'n_radial_increment', 0),
            'n_angular_increment': getattr(config, 'n_angular_increment', 0),
            'use_analytic': use_analytic,
            'seed': config.seed,
            'lambda_penalty': getattr(config, 'lambda_penalty', 1.0),
            'use_custom_initial_condition': getattr(config, 'use_custom_initial_condition', False),
            'initial_condition_path': getattr(config, 'initial_condition_path', None)
        },
        'final_mesh_stats': {
            'n_vertices': int(final_result['mesh_stats']['n_vertices']),
            'n_triangles': int(final_result['mesh_stats']['n_triangles']),
            'total_area': float(final_result['mesh_stats']['total_area']),
            'theoretical_area': float(final_result['mesh_stats']['theoretical_area']),
            'min_triangle_area': float(final_result['mesh_stats']['min_triangle_area']),
            'max_triangle_area': float(final_result['mesh_stats']['max_triangle_area']),
            'mean_triangle_area': float(final_result['mesh_stats']['mean_triangle_area']),
            'r_inner': float(final_result['mesh_stats']['r_inner']),
            'r_outer': float(final_result['mesh_stats']['r_outer']),
            'n_radial': int(final_result['mesh_stats']['n_radial']),
            'n_angular': int(final_result['mesh_stats']['n_angular'])
        },
        'final_epsilon': float(final_result['epsilon']),
        'final_energy': float(final_result['energy']),
        'final_iterations': int(final_result['iterations']),
        'run_time_seconds': float(final_result['time']),
        'success': bool(final_result['success']),
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
    
    print(f"Ring partition optimization complete. See {logfile_path} for detailed logs.\n")
    print(f"Results saved in: {outdir}")
    print(f"Use optimization_analyzer.py to analyze the results.")
    
    return results

def main():
    """
    Main function for ring partition optimization.
    
    This implements the Γ-convergence approach described in the paper:
    "Partitions of Minimal Length on Manifolds" by Bogosel and Oudet.
    """
    parser = argparse.ArgumentParser(description='Ring partition optimization using Γ-convergence')
    parser.add_argument('--input', type=str, help='Path to input YAML file with parameters')
    parser.add_argument('--solution-dir', type=str, help='Directory for storing solution files')
    parser.add_argument('--initial-condition', type=str, help='Path to .h5 file containing initial condition')
    args = parser.parse_args()

    # Setup logging - we'll set up file logging in optimize_partition_ring
    setup_logging(log_level='INFO', log_to_console=True, log_to_file=False)
    logger = get_logger(__name__)
    logger.info("Starting ring partition optimization")

    # Load configuration
    if args.input:
        print(f"\nLoading parameters from {args.input}")
        with open(args.input, 'r') as f:
            params = yaml.safe_load(f)
        
        # Override initial condition path if provided via command line
        if args.initial_condition:
            params['use_custom_initial_condition'] = True
            params['initial_condition_path'] = args.initial_condition
        
        config = Config(params)
    else:
        # Use default configuration
        config = Config()
        if args.initial_condition:
            config.use_custom_initial_condition = True
            config.initial_condition_path = args.initial_condition

    # Run the optimization
    optimize_partition_ring(
        config=config,
        solution_dir=args.solution_dir
    )

if __name__ == "__main__":
    main() 