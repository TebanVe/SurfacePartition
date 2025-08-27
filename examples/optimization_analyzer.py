#!/usr/bin/env python3
"""
Analysis and visualization tool for ring partition optimization results.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import yaml
import argparse
import glob
from typing import Dict, List, Optional, Tuple
import logging
import re

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from logging_config import get_logger
from surfaces.ring import RingMeshProvider

def load_pyslsqp_internal_data(hdf5_file_path: str) -> Optional[Dict]:
	"""
	Load internal optimization data from HDF5 file and return arrays plus
	iteration indices parsed from group names.
	"""
	if not os.path.exists(hdf5_file_path):
		return None
	try:
		with h5py.File(hdf5_file_path, 'r') as f:
			data: Dict[str, np.ndarray] = {}
			iter_keys = [k for k in f.keys() if k.startswith('iter_')]
			iter_keys.sort(key=lambda x: int(x.split('_')[1]))
			iters = [int(k.split('_')[1]) for k in iter_keys]
			x_list: List[np.ndarray] = []
			grad_list: List[np.ndarray] = []
			obj_list: List[float] = []
			constraints_list: List[np.ndarray] = []
			ismajor_list: List[bool] = []
			for k in iter_keys:
				g = f[k]
				if 'x' in g:
					x_list.append(g['x'][:])
				if 'gradient' in g:
					grad_list.append(g['gradient'][:])
				if 'objective' in g:
					obj_list.append(g['objective'][()])
				if 'constraints' in g:
					constraints_list.append(g['constraints'][:])
				if 'ismajor' in g:
					ismajor_list.append(bool(g['ismajor'][()]))
				else:
					ismajor_list.append(True)
			if x_list:
				data['x'] = np.array(x_list)
			if grad_list:
				data['gradient'] = np.array(grad_list)
			if obj_list:
				data['objective'] = np.array(obj_list)
			if constraints_list:
				data['constraints'] = np.array(constraints_list)
			if ismajor_list:
				data['ismajor'] = np.array(ismajor_list)
			data['iters'] = np.array(iters, dtype=int)
			return data
	except Exception as e:
		print(f"Error loading HDF5 file {hdf5_file_path}: {e}")
		return None

def parse_pyslsqp_summary_file(summary_file_path: str) -> Optional[Dict]:
    """
    Parse PySLSQP summary file to extract optimization metrics.
    
    Args:
        summary_file_path: Path to summary file
        
    Returns:
        Dictionary containing parsed metrics or None if file doesn't exist
    """
    if not os.path.exists(summary_file_path):
        return None
    
    try:
        with open(summary_file_path, 'r') as f:
            lines = f.readlines()
        
        # Skip header line and filter data lines
        data_lines = [line.strip() for line in lines if line.strip() and not line.startswith('MAJOR')]
        
        energies = []
        grad_norms = []
        constraints = []
        steps = []
        feas = []
        opt = []
        
        for line in data_lines:
            parts = line.split()
            if len(parts) >= 8:  # MAJOR NFEV NGEV OBJFUN GNORM CNORM FEAS OPT STEP
                energies.append(float(parts[3]))  # OBJFUN
                grad_norms.append(float(parts[4]))  # GNORM
                constraints.append(float(parts[5]))  # CNORM
                feas.append(float(parts[6]))  # FEAS
                opt.append(float(parts[7]))  # OPT
                steps.append(float(parts[8]))  # STEP
        
        return {
            'energies': energies,
            'grad_norms': grad_norms,
            'constraints': constraints,
            'steps': steps,
            'feas': feas,
            'opt': opt
        }
        
    except Exception as e:
        print(f"Error parsing summary file {summary_file_path}: {e}")
        return None

def load_pyslsqp_optimization_data(results: List[Dict], logger=None) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[int]]:
    """
    Load and aggregate optimization data from summary files across all refinement levels.
    Returns (energies, grad_norms, cnorms, feas, steps, level_boundaries).
    """
    if logger is None:
        logger = get_logger(__name__)
    
    all_energies = []
    all_grad_norms = []
    all_constraints = []
    all_steps = []
    all_feas = []
    level_boundaries = []
    total_iters = 0
    
    for result in results:
        summary_file = result.get('summary_file')
        if summary_file and os.path.exists(summary_file):
            if logger:
                logger.info(f"Loading optimization data from {summary_file}")
            
            data = parse_pyslsqp_summary_file(summary_file)
            if data:
                # Extract data directly from summary file
                level_energies = data['energies']
                level_grad_norms = data['grad_norms']
                level_constraints = data['constraints']
                level_steps = data['steps']
                level_feas = data.get('feas', [])
                
                all_energies.extend(level_energies)
                all_grad_norms.extend(level_grad_norms)
                all_constraints.extend(level_constraints)
                all_steps.extend(level_steps)
                all_feas.extend(level_feas)
                
                # Update iteration count and boundaries
                num_iters = len(level_energies)
                total_iters += num_iters
                level_boundaries.append(total_iters)
                
                if logger:
                    logger.debug(f"  Loaded {num_iters} iterations")
            else:
                if logger:
                    logger.warning(f"Failed to load data from {summary_file}")
                # Add boundary for empty level
                level_boundaries.append(total_iters)
        else:
            if logger:
                logger.warning(f"Missing or invalid summary_file")
            # Add boundary for missing level
            level_boundaries.append(total_iters)
    
    if logger:
        logger.info(f"Loaded total of {total_iters} iterations")
    
    return all_energies, all_grad_norms, all_constraints, all_feas, all_steps, level_boundaries

def extract_constraint_evolution_from_pyslsqp_data(results: List[Dict], n_partitions: int, 
													 logger=None, major_only: bool = False) -> Dict:
	"""
	Extract constraint evolution data and iteration indices.
	Returns dict with cnorm, feas, areas, unity, and area_iters (global indices).
	"""
	if logger is None:
		logger = get_logger(__name__)
	cnorm_evolution: List[float] = []
	feas_evolution: List[float] = []
	area_evolution: List[np.ndarray] = []
	unity_evolution: List[np.ndarray] = []
	area_iters: List[int] = []
	for result in results:
		internal_data_file = result.get('internal_data_file')
		start_global = int(result.get('start_index_global', 0))
		if internal_data_file and os.path.exists(internal_data_file):
			data = load_pyslsqp_internal_data(internal_data_file)
			if data is None or 'x' not in data:
				continue
			v_level = result.get('v')
			if v_level is None:
				logger.warning("No 'v' vector found in result, skipping")
				continue
			# Select series
			x_all = data['x']
			iters_local = data.get('iters')
			if major_only and 'ismajor' in data:
				major_idx = np.where(data['ismajor'])[0]
				x_all = x_all[major_idx]
				if iters_local is not None:
					iters_local = iters_local[major_idx]
			# Compute
			for i, x in enumerate(x_all):
				N = len(x) // n_partitions
				if len(x) != N * n_partitions:
					logger.warning(f"Data size {len(x)} is not divisible by n_partitions {n_partitions}")
					continue
				phi = x.reshape(N, n_partitions)
				areas = v_level @ phi
				area_evolution.append(areas.copy())
				if iters_local is not None:
					area_iters.append(int(start_global + int(iters_local[i])))
				# unity violation per vertex (kept for potential homogeneous plotting)
				unity_violations = np.sum(phi, axis=1) - 1.0
				unity_evolution.append(unity_violations.copy())
			# constraints/feas if present
			if 'constraints' in data:
				c_all = data['constraints']
				if major_only and 'ismajor' in data:
					c_all = c_all[major_idx]
				for j in range(min(len(c_all), len(x_all))):
					cnorm_evolution.append(float(np.linalg.norm(c_all[j])))
					feas_evolution.append(float(np.max(np.abs(c_all[j]))))
	if logger:
		logger.info(f"Extracted total of {len(area_evolution)} area snapshots")
	return {
		'cnorm': cnorm_evolution,
		'feas': feas_evolution,
		'areas': area_evolution,
		'unity': unity_evolution,
		'area_iters': area_iters,
	}

def compute_unity_last_level(internal_data_file: str, n_partitions: int, major_only: bool = False, logger=None) -> Optional[np.ndarray]:
    """
    Compute partition-of-unity violations for the last (finest) level only.
    Returns an array of shape (n_iters_last, N_last) with per-vertex violations Σ u_i - 1 at each iterate.
    """
    if logger is None:
        logger = get_logger(__name__)
    data = load_pyslsqp_internal_data(internal_data_file)
    if not data or 'x' not in data:
        if logger:
            logger.warning(f"No 'x' data in {internal_data_file} for last-level unity computation")
        return None
    x_all = data['x']
    if major_only and 'ismajor' in data:
        major_idx = np.where(data['ismajor'])[0]
        x_all = x_all[major_idx]
    # Determine N from vector length
    L = x_all.shape[1]
    if L % n_partitions != 0:
        logger.warning(f"Vector length {L} not divisible by n_partitions {n_partitions}; skipping unity plot")
        return None
    N = L // n_partitions
    unity_list = []
    for x in x_all:
        phi = x.reshape(N, n_partitions)
        unity_violation = np.sum(phi, axis=1) - 1.0
        unity_list.append(unity_violation)
    if not unity_list:
        return None
    return np.vstack(unity_list)

def plot_refinement_optimization_metrics(energies: List[float], grad_norms: List[float], 
                                       constraints: List[float], steps: List[float], 
                                       level_boundaries: List[int], 
                                       save_path: str = 'refinement_optimization_metrics.png',
                                       n_partitions: Optional[int] = None, 
                                       n_radial_info: Optional[str] = None,
                                       n_angular_info: Optional[str] = None,
                                       lambda_penalty: Optional[float] = None,
                                       seed: Optional[int] = None,
                                       use_analytic: Optional[bool] = None,
                                       title_override: Optional[str] = None):
    """
    Create 2x2 grid of optimization metrics plots.
    
    Args:
        energies: List of energy values
        grad_norms: List of gradient norm values
        constraints: List of constraint violation values
        steps: List of step size values
        level_boundaries: List of level boundary indices
        save_path: Path to save the plot
        n_partitions: Number of partitions
        n_radial_info: Radial mesh info
        n_angular_info: Angular mesh info
        lambda_penalty: Lambda penalty value
        seed: Random seed
        use_analytic: Whether analytic gradients were used
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot energy convergence
    axes[0, 0].plot(energies, 'b-', linewidth=2)
    axes[0, 0].set_title('Energy Convergence')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Energy')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot gradient norm convergence
    axes[0, 1].plot(grad_norms, 'r-', linewidth=2)
    axes[0, 1].set_title('Gradient Norm Convergence')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Gradient Norm')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot constraint violation convergence
    axes[1, 0].plot(constraints, 'g-', linewidth=2)
    axes[1, 0].set_title('Constraint Violation Convergence')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Constraint Violation')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot step size evolution
    axes[1, 1].plot(steps, 'purple', linewidth=2)
    axes[1, 1].set_title('Step Size Evolution')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Step Size')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add level boundaries if provided
    if level_boundaries:
        for boundary in level_boundaries:
            for ax in axes.flat:
                ax.axvline(x=boundary, color='k', linestyle='--', alpha=0.5)
    
    # Title
    if title_override:
        fig.suptitle(title_override, fontsize=14)
    else:
        title_parts = []
        if n_partitions:
            title_parts.append(f"n_partitions={n_partitions}")
        if n_radial_info:
            title_parts.append(f"n_radial={n_radial_info}")
        if n_angular_info:
            title_parts.append(f"n_angular={n_angular_info}")
        if lambda_penalty is not None:
            title_parts.append(f"lambda={lambda_penalty}")
        if seed:
            title_parts.append(f"seed={seed}")
        if use_analytic is not None:
            title_parts.append(f"analytic_gradients={'yes' if use_analytic else 'no'}")
        if title_parts:
            fig.suptitle(f"PySLSQP Optimization Metrics: {', '.join(title_parts)}", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved optimization metrics plot to: {save_path}")

def plot_constraint_evolution(constraint_data: Dict, level_boundaries: List[int],
                            save_path: str = 'constraint_evolution.png',
                            n_partitions: Optional[int] = None,
                            n_radial_info: Optional[str] = None,
                            n_angular_info: Optional[str] = None,
                            lambda_penalty: Optional[float] = None,
                            seed: Optional[int] = None,
                            use_analytic: Optional[bool] = None,
                            max_vertices_plot: int = 50,
                            unity_last_level: Optional[np.ndarray] = None,
                            unity_last_start: Optional[int] = None,
                            unity_last_iters: Optional[np.ndarray] = None,
                            theoretical_total_area: Optional[float] = None,
                            title_override: Optional[str] = None,
                            logger=None):
    """
    Create 2x2 grid of constraint evolution plots.
    
    Args:
        constraint_data: Dictionary containing constraint evolution data
        level_boundaries: List of level boundary indices
        save_path: Path to save the plot
        n_partitions: Number of partitions
        n_radial_info: Radial mesh info
        n_angular_info: Angular mesh info
        lambda_penalty: Lambda penalty value
        seed: Random seed
        use_analytic: Whether analytic gradients were used
        max_vertices_plot: Maximum number of vertices to plot
        unity_last_level: If provided, 2D array (n_iters_last, N_last) of last-level unity violations
        unity_last_start: If provided, iteration offset to align last-level unity on global axis
        logger: Logger instance
    """
    if logger is None:
        logger = get_logger(__name__)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract data
    cnorm_evolution = constraint_data.get('cnorm', [])
    feas_evolution = constraint_data.get('feas', [])
    area_evolution = constraint_data.get('areas', [])
    unity_evolution = constraint_data.get('unity', [])
    area_iters = np.array(constraint_data.get('_area_iters', [])) if constraint_data.get('_area_iters') is not None else None
    
    # Check if we have data
    if len(area_evolution) == 0:
        plt.text(0.5, 0.5, 'No constraint evolution data available', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Constraint Evolution - No Data')
        plt.savefig(save_path)
        plt.close()
        return
    
    # Convert to numpy arrays
    area_evolution = np.array(area_evolution)
    
    # Handle unity_evolution which may have inhomogeneous shapes
    try:
        unity_evolution = np.array(unity_evolution)
        unity_evolution_homogeneous = True
    except ValueError as e:
        if "inhomogeneous shape" in str(e):
            unity_evolution_homogeneous = False
            if logger:
                logger.warning("Unity evolution arrays have different shapes. Skipping unity violation plot.")
        else:
            raise e
    
    # Plot constraint norm convergence
    if len(cnorm_evolution) > 0:
        axes[0, 0].plot(cnorm_evolution, 'b-', linewidth=2, label='CNORM')
        axes[0, 0].set_title('Constraint Norm Convergence')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Constraint Norm')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
    else:
        axes[0, 0].text(0.5, 0.5, 'No CNORM data available', 
                       ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Constraint Norm Convergence')
    
    # Plot feasibility convergence
    if len(feas_evolution) > 0:
        axes[0, 1].plot(feas_evolution, 'r-', linewidth=2, label='FEAS')
        axes[0, 1].set_title('Feasibility Convergence')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Feasibility')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
    else:
        axes[0, 1].text(0.5, 0.5, 'No FEAS data available', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Feasibility Convergence')
    
    # Plot area evolution with sparse x
    if isinstance(area_evolution, list) and len(area_evolution) > 0:
        area_arr = np.array(area_evolution)
        n_partitions_actual = area_arr.shape[1]
        
        # Target area: use theoretical total area if provided; else skip line
        if theoretical_total_area is not None:
            target_area = theoretical_total_area / n_partitions_actual
            axes[1, 0].axhline(y=target_area, color='k', linestyle='-', label='Target Area')
        
        # Build x-axis
        if isinstance(area_iters, np.ndarray) and area_iters.size == area_arr.shape[0]:
            xs = area_iters
        else:
            # Fallback: regular spacing
            xs = np.arange(area_arr.shape[0])
        
        # Plot each partition's area
        for i in range(n_partitions_actual):
            axes[1, 0].plot(xs, area_arr[:, i], linestyle='--', alpha=0.7, 
                           label=f'Partition {i+1}')
        
        axes[1, 0].set_title('Area Evolution')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Area')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, 'No area data available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Area Evolution')
    
    # Plot partition unity violations
    unity_last_level_data = unity_last_level
    if unity_last_level_data is not None and isinstance(unity_last_level_data, np.ndarray):
        # Plot only last-level unity, aligned to global iteration axis
        n_vertices = unity_last_level_data.shape[1]
        if n_vertices > max_vertices_plot:
            vertex_indices = np.linspace(0, n_vertices-1, max_vertices_plot, dtype=int)
            sampled_unity = unity_last_level_data[:, vertex_indices]
        else:
            sampled_unity = unity_last_level_data
        x0 = unity_last_start or 0
        xs = np.arange(sampled_unity.shape[0]) + x0
        axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.5, label='Target (Unity)')
        for i in range(sampled_unity.shape[1]):
            axes[1, 1].plot(xs, sampled_unity[:, i], linestyle=':', alpha=0.7)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Unity Violation')
        axes[1, 1].set_title('Partition Unity Violations (last level)')
        axes[1, 1].grid(True, alpha=0.3)
    elif unity_evolution_homogeneous and unity_evolution.shape[0] > 0:
        n_vertices = unity_evolution.shape[1]
        
        # Sample vertices if too many
        if n_vertices > max_vertices_plot:
            vertex_indices = np.linspace(0, n_vertices-1, max_vertices_plot, dtype=int)
            sampled_unity = unity_evolution[:, vertex_indices]
        else:
            sampled_unity = unity_evolution
        
        # Plot target line at 0
        axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.5, label='Target (Unity)')
        
        # Plot sampled vertices
        for i in range(sampled_unity.shape[1]):
            axes[1, 1].plot(sampled_unity[:, i], linestyle=':', alpha=0.7)
        
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Unity Violation')
        axes[1, 1].set_title('Partition Unity Violations')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        if not unity_evolution_homogeneous:
            axes[1, 1].text(0.5, 0.5, 'Unity violations not available\n(different mesh sizes)', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        else:
            axes[1, 1].text(0.5, 0.5, 'No unity violation data available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Partition Unity Violations')
    
    # Add level boundaries if provided
    for boundary in level_boundaries:
        for ax in axes.flat:
            ax.axvline(x=boundary, color='k', linestyle='--', alpha=0.5)
    
    # Title
    if title_override:
        fig.suptitle(title_override, fontsize=16)
    else:
        title_parts = []
        if n_partitions:
            title_parts.append(f"n_partitions={n_partitions}")
        if n_radial_info:
            title_parts.append(f"n_radial={n_radial_info}")
        if n_angular_info:
            title_parts.append(f"n_angular={n_angular_info}")
        if lambda_penalty is not None:
            title_parts.append(f"lambda={lambda_penalty}")
        if seed:
            title_parts.append(f"seed={seed}")
        if use_analytic is not None:
            title_parts.append(f"analytic_gradients={'yes' if use_analytic else 'no'}")
        if title_parts:
            fig.suptitle(f"PySLSQP Constraint Evolution: {', '.join(title_parts)}", fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close()

def analyze_optimization_run(results_dir: str, output_dir: str = None):
    """
    Analyze an optimization run by loading data and generating plots.
    """
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
    logger = get_logger(__name__)
    
    if output_dir is None:
        output_dir = results_dir
    
    logger.info(f"Analyzing optimization run in: {results_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load metadata
    metadata_file = os.path.join(results_dir, 'metadata.yaml')
    if not os.path.exists(metadata_file):
        logger.error(f"Metadata file not found: {metadata_file}")
        return
    
    with open(metadata_file, 'r') as f:
        metadata = yaml.safe_load(f)
    
    logger.info(f"Loaded metadata for run: {metadata.get('datetime', 'unknown')}")
    
    # Read per-level info directly from metadata
    levels_meta = metadata.get('levels') if isinstance(metadata.get('levels'), list) else []
    if not levels_meta:
        logger.error("No level metadata found; cannot analyze run")
        return
    # Sort by explicit level index
    levels_meta_sorted = sorted(levels_meta, key=lambda x: int(x.get('level', 0)))
    
    # Load mesh parameters from metadata
    mesh_params = metadata.get('final_mesh_stats', {})
    n_radial = mesh_params.get('n_radial', 8)
    n_angular = mesh_params.get('n_angular', 16)
    r_inner = mesh_params.get('r_inner', 0.5)
    r_outer = mesh_params.get('r_outer', 1.0)
    
    # Extract constraint evolution data using metadata exclusively
    surface = metadata.get('input_parameters', {}).get('surface')
    n_partitions = int(metadata.get('input_parameters', {}).get('n_partitions', metadata.get('n_partitions', 2)))
    results = []
    theoretical_total_area = metadata.get('theoretical_total_area')
    provider = RingMeshProvider(n_radial, n_angular, r_inner, r_outer)
    if theoretical_total_area is None and hasattr(provider, 'theoretical_total_area'):
        theoretical_total_area = provider.theoretical_total_area()
    for lm in levels_meta_sorted:
        nr_eff = int(lm.get('nr', lm.get('nt', lm.get('v1', n_radial))))
        na_eff = int(lm.get('na', lm.get('np', lm.get('v2', n_angular))))
        provider.set_resolution(nr_eff, na_eff)
        mesh_level = provider.build()
        mesh_level.compute_matrices()
        v_level = mesh_level.v
        internal_data_file = lm.get('files', {}).get('internal_data')
        start_global = int(lm.get('iters', {}).get('start_index_global', 0))
        if internal_data_file and os.path.exists(internal_data_file):
            results.append({'internal_data_file': internal_data_file, 'v': v_level, 'N_meta': int(lm.get('N', len(v_level))), 'start_index_global': start_global})
     
    constraint_data = extract_constraint_evolution_from_pyslsqp_data(results, n_partitions, logger)
    # Compute last-level unity violations and iteration offset for plotting
    last_level_meta = levels_meta_sorted[-1]
    last_internal_file = last_level_meta.get('files', {}).get('internal_data')
    unity_last_level = None
    unity_last_iters_local = None
    if last_internal_file and os.path.exists(last_internal_file):
        unity_last_level = compute_unity_last_level(last_internal_file, n_partitions, major_only=False, logger=logger)
        unity_last_iters_local = load_pyslsqp_internal_data(last_internal_file).get('iters')
    
    # Load optimization data from summary files listed in metadata
    summary_files = []
    for lm in levels_meta_sorted:
        sfile = lm.get('files', {}).get('summary')
        if sfile and os.path.exists(sfile):
            summary_files.append(sfile)
    if not summary_files:
        logger.error("No summary files found in metadata")
        return
    energies, grad_norms, cnorms, feas_series, steps, level_boundaries = load_pyslsqp_optimization_data(
        [{'summary_file': f} for f in summary_files], logger
    )
    # Prefer summary-derived CNORM/FEAS series when available
    if cnorms:
        constraint_data['cnorm'] = cnorms
    if feas_series:
        constraint_data['feas'] = feas_series
    
    # Determine global iteration offset for last level from metadata if available
    unity_last_start = int(last_level_meta.get('iters', {}).get('start_index_global', (level_boundaries[-2] if len(level_boundaries) > 1 else 0)))
    
    # Create optimization metrics plot
    # Build common title from metadata when available
    title_labels = metadata.get('input_parameters', {}).get('resolution_labels') or ['v1', 'v2']
    label1 = title_labels[0]
    label2 = title_labels[1] if len(title_labels) > 1 else 'v2'
    last_level = levels_meta_sorted[-1] if isinstance(levels_meta_sorted, list) and levels_meta_sorted else {}
    var1_val = int(last_level.get(label1, mesh_params.get('n_radial', 0)))
    var2_val = int(last_level.get(label2, mesh_params.get('n_angular', 0)))
    optimizer_name = metadata.get('optimizer') or 'PGD'
    lam = metadata.get('input_parameters', {}).get('lambda_penalty')
    seed = metadata.get('input_parameters', {}).get('seed')
    use_analytic_flag = metadata.get('input_parameters', {}).get('use_analytic')
    from src.plot_utils import build_plot_title
    metrics_title = build_plot_title(optimizer_name, surface, label1, var1_val, label2, var2_val, lam, seed, use_analytic_flag, prefix='Optimization Metrics')
    plot_refinement_optimization_metrics(
        energies, grad_norms, cnorms, steps, level_boundaries,
        save_path=os.path.join(output_dir, 'refinement_optimization_metrics.png'),
        use_analytic=metadata.get('input_parameters', {}).get('use_analytic'),
        title_override=metrics_title
    )
    
    # Create constraint evolution plot
    constraint_title = build_plot_title(optimizer_name, surface, label1, var1_val, label2, var2_val, lam, seed, use_analytic_flag, prefix='Constraint Evolution')
    plot_constraint_evolution(
        constraint_data, level_boundaries,
        save_path=os.path.join(output_dir, 'constraint_evolution.png'),
        n_partitions=n_partitions,
        n_radial_info=mesh_params.get('n_radial'),
        n_angular_info=mesh_params.get('n_angular'),
        lambda_penalty=metadata.get('input_parameters', {}).get('lambda_penalty'),
        seed=metadata.get('input_parameters', {}).get('seed'),
        use_analytic=metadata.get('input_parameters', {}).get('use_analytic'),
        unity_last_level=unity_last_level,
        unity_last_start=unity_last_start,
        unity_last_iters=unity_last_iters_local,
        theoretical_total_area=theoretical_total_area,
        title_override=constraint_title,
        logger=logger
    )
    
    logger.info(f"Analysis complete. Plots saved in: {output_dir}")

def main():
    """
    Main function for optimization analysis.
    """
    parser = argparse.ArgumentParser(description='Analyze ring partition optimization results')
    parser.add_argument('--results-dir', type=str, required=True, 
                       help='Directory containing optimization results')
    parser.add_argument('--output-dir', type=str, 
                       help='Directory to save analysis plots (defaults to results directory)')
    parser.add_argument('--pattern', type=str, 
                       help='Pattern to match multiple result directories')
    args = parser.parse_args()
    
    if args.pattern:
        # Analyze multiple runs matching pattern
        import glob
        pattern = os.path.join(args.results_dir, f"*{args.pattern}*")
        result_dirs = glob.glob(pattern)
        
        print(f"Found {len(result_dirs)} result directories matching pattern: {args.pattern}")
        for result_dir in result_dirs:
            if os.path.isdir(result_dir):
                print(f"\nAnalyzing: {result_dir}")
                analyze_optimization_run(result_dir, args.output_dir)
    else:
        # Analyze single run
        analyze_optimization_run(args.results_dir, args.output_dir)

if __name__ == "__main__":
    main() 