import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import logging
import os
import datetime
from functools import wraps

# Try to import PySLSQP, fall back to a mock if not available
try:
    import pyslsqp
    PYSLSQP_AVAILABLE = True
    from pyslsqp.postprocessing import print_dict_as_table
except ImportError:
    PYSLSQP_AVAILABLE = False
    print("Warning: PySLSQP not available. Using mock implementation for testing.")

# Import our logging system
try:
    from .logging_config import get_logger, log_performance, log_performance_conditional
except ImportError:
    # Fallback for when running as script
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    from logging_config import get_logger, log_performance, log_performance_conditional

class RefinementTriggered(Exception):
    """Exception raised when mesh refinement is triggered during optimization."""
    pass

class PySLSQPOptimizer:
    """
    PySLSQP optimizer for ring partition optimization using Γ-convergence.
    
    This class implements the optimization method described in "Partitions of Minimal 
    Length on Manifolds" by Bogosel and Oudet, adapted for a 2D ring (annulus) domain.
    
    The energy functional is:
        J_ε(u) = ε ∫_Ω |∇u|² + (1/ε) ∫_Ω u²(1-u)²
    
    For n partitions, we minimize:
        E = Σ_{i=1}^n J_ε(u_i) + penalty terms
    
    Subject to constraints:
        - Partition constraint: Σ_{i=1}^n u_i = 1 (everywhere)
        - Area constraint: ∫_Ω u_i = A/n (equal areas)
    """
    
    def __init__(self, 
                 K: np.ndarray,  # Stiffness matrix
                 M: np.ndarray,  # Mass matrix
                 v: np.ndarray,  # Mass matrix column sums (v = 1ᵀM)
                 n_partitions: int,
                 epsilon: float,  # Interface width parameter
                 r_inner: float,  # Ring inner radius
                 r_outer: float,  # Ring outer radius
                 lambda_penalty: float = 1.0,
                 starget: Optional[float] = None,
                 refine_patience: int = 30,
                 refine_delta_energy: float = 1e-4,
                 refine_grad_tol: float = 1e-2,
                 refine_constraint_tol: float = 1e-2,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the PySLSQP optimizer for ring partition optimization.
        
        Args:
            K: Stiffness matrix for gradient term
            M: Mass matrix for area term
            v: Vector of mass matrix column sums (v = 1ᵀM)
            n_partitions: Number of partitions
            epsilon: Interface width parameter
            r_inner: Ring inner radius
            r_outer: Ring outer radius
            lambda_penalty: Initial penalty weight for constant functions
            starget: Target standard deviation (if None, computed from area)
            refine_patience: Patience for hybrid refinement trigger
            refine_delta_energy: Energy threshold for refinement
            refine_grad_tol: Gradient tolerance for refinement
            refine_constraint_tol: Constraint tolerance for refinement
            logger: Logger instance (if None, creates one)
        """
        if not PYSLSQP_AVAILABLE:
            raise ImportError("PySLSQP is not available. Please install it first.")
        
        # Validate input parameters
        self._validate_inputs(K, M, v, n_partitions, epsilon, r_inner, r_outer)
        
        # Store matrices and parameters
        self.K = K
        self.M = M
        self.v = v
        self.n_partitions = n_partitions
        self.epsilon = epsilon
        self.r_inner = r_inner
        self.r_outer = r_outer
        self.lambda_penalty = lambda_penalty
        
        # Calculate theoretical total area of ring
        self.theoretical_total_area = np.pi * (r_outer**2 - r_inner**2)
        
        # Get discretized mesh statistics
        self.total_area = np.sum(v)
        
        # Compute target standard deviation if not provided
        if starget is None:
            normalized_area = 1.0/n_partitions
            self.starget = np.sqrt(normalized_area * (1 - normalized_area))
        else:
            self.starget = starget
            
        # Refinement parameters
        self.refine_patience = refine_patience
        self.refine_delta_energy = refine_delta_energy
        self.refine_grad_tol = refine_grad_tol
        self.refine_constraint_tol = refine_constraint_tol
        
        # Initialize logging
        if logger is None:
            self.logger = get_logger(__name__)
        else:
            self.logger = logger
            
        # Initialize optimization log
        self.log = {
            'iterations': [],
            'energy_changes': [],
            'warnings': [],
            'area_evolution': []
        }
        
        self.logger.info(f"PySLSQPOptimizer initialized:")
        self.logger.info(f"  Ring: r_inner={r_inner:.3f}, r_outer={r_outer:.3f}")
        self.logger.info(f"  Partitions: {n_partitions}")
        self.logger.info(f"  Epsilon: {epsilon:.2e}")
        self.logger.info(f"  Theoretical area: {self.theoretical_total_area:.6f}")
        self.logger.info(f"  Discretized area: {self.total_area:.6f}")
    
    def _validate_inputs(self, K, M, v, n_partitions, epsilon, r_inner, r_outer):
        """Validate input parameters."""
        if n_partitions < 2:
            raise ValueError("n_partitions must be at least 2")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if r_inner <= 0 or r_outer <= 0:
            raise ValueError("Ring radii must be positive")
        if r_inner >= r_outer:
            raise ValueError("r_inner must be less than r_outer")
        if K.shape != M.shape:
            raise ValueError("K and M must have the same shape")
        if len(v) != K.shape[0]:
            raise ValueError("v must have length equal to matrix dimension")
        if K.shape[0] != K.shape[1]:
            raise ValueError("K and M must be square matrices")
    
    def validate_initial_condition(self, x0: np.ndarray) -> bool:
        """
        Validate that the initial condition satisfies basic constraints.
        
        Args:
            x0: Initial condition vector
            
        Returns:
            True if valid, False otherwise
        """
        N = len(self.v)
        n = self.n_partitions
        
        # Check dimensions
        if len(x0) != N * n:
            self.logger.error(f"Initial condition dimension mismatch: got {len(x0)}, expected {N * n}")
            return False
        
        # Reshape to check constraints
        phi = x0.reshape(N, n)
        
        # Check bounds: 0 ≤ x ≤ 1
        if np.any(phi < 0) or np.any(phi > 1):
            self.logger.error("Initial condition violates bounds: values must be in [0, 1]")
            return False
        
        # Check partition constraint: Σ u_i = 1 (everywhere)
        row_sums = np.sum(phi, axis=1)
        partition_violation = np.max(np.abs(row_sums - 1.0))
        if partition_violation > 1e-6:
            self.logger.warning(f"Initial condition violates partition constraint: max violation = {partition_violation:.2e}")
            return False
        
        self.logger.info("✓ Initial condition validation passed")
        return True
    
    def generate_initial_condition(self, method: str = "random") -> np.ndarray:
        """
        Generate a valid initial condition.
        
        Args:
            method: Method to generate initial condition ("random", "uniform", "radial")
            
        Returns:
            Valid initial condition vector
        """
        N = len(self.v)
        n = self.n_partitions
        
        if method == "random":
            # Generate random initial condition
            x0 = np.random.rand(N * n)
            # Normalize to satisfy partition constraint
            for i in range(N):
                row_sum = np.sum(x0[i::N])
                if row_sum > 0:
                    x0[i::N] /= row_sum
                else:
                    # If all zeros, set to uniform
                    x0[i::N] = 1.0 / n
                    
        elif method == "uniform":
            # Generate uniform initial condition (equal partitions)
            x0 = np.ones(N * n) / n
            
        elif method == "radial":
            # Generate radial initial condition (partitions based on radial position)
            phi = np.zeros((N, n))
            
            # Get vertex coordinates from mesh (if available)
            # For now, use a simple radial partitioning
            for i in range(N):
                # Simple radial partitioning
                partition_idx = (i % n)
                phi[i, partition_idx] = 1.0
            
            x0 = phi.flatten()
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.logger.info(f"Generated initial condition using method: {method}")
        return x0
    
    def process_initial_condition(self, x0: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Process and clean the initial condition.
        
        Args:
            x0: Initial condition vector
            normalize: Whether to normalize to satisfy partition constraint
            
        Returns:
            Processed initial condition
        """
        N = len(self.v)
        n = self.n_partitions
        
        # Ensure correct shape
        if len(x0) != N * n:
            raise ValueError(f"Initial condition dimension mismatch: got {len(x0)}, expected {N * n}")
        
        # Reshape for processing
        phi = x0.reshape(N, n).copy()
        
        # Ensure bounds: 0 ≤ x ≤ 1
        phi = np.clip(phi, 0, 1)
        
        if normalize:
            # Normalize to satisfy partition constraint: Σ u_i = 1
            for i in range(N):
                row_sum = np.sum(phi[i, :])
                if row_sum > 0:
                    phi[i, :] /= row_sum
                else:
                    # If all zeros, set to uniform
                    phi[i, :] = 1.0 / n
        
        processed_x0 = phi.flatten()
        
        self.logger.info("✓ Initial condition processed")
        return processed_x0
    
    def compute_energy(self, x: np.ndarray) -> float:
        """
        Compute the Γ-convergence energy functional.
        
        The energy is:
            J_ε(u) = ε ∫_Ω |∇u|² + (1/ε) ∫_Ω u²(1-u)²
        
        For n partitions, we minimize:
            E = Σ_{i=1}^n J_ε(u_i) + penalty terms
        
        Args:
            x: Optimization variable (flattened n_partitions × N)
            
        Returns:
            Total energy value
        """
        N = len(self.v)
        n = self.n_partitions
        
        # Reshape x to matrix form: (N, n_partitions)
        phi = x.reshape(N, n)
        
        # Compute energy for each partition
        total_energy = 0.0
        
        for i in range(n):
            phi_i = phi[:, i]
            
            # Gradient term: ε ∫_Ω |∇u|²
            gradient_term = self.epsilon * float(phi_i.T @ (self.K @ phi_i))
            
            # Interface term: (1/ε) ∫_Ω u²(1-u)²
            interface_vec = phi_i**2 * (1 - phi_i)**2
            interface_term = (1/self.epsilon) * float(interface_vec.T @ (self.M @ interface_vec))
            
            # Add to total energy
            total_energy += gradient_term + interface_term
        
        # Add penalty term for constant functions (optional)
        if self.lambda_penalty > 0:
            penalty = 0.0
            for i in range(n):
                phi_i = phi[:, i]
                # Penalty for functions that are too constant
                mean_val = np.mean(phi_i)
                variance = np.var(phi_i)
                penalty += self.lambda_penalty * (1.0 - variance / (mean_val * (1 - mean_val) + 1e-8))
            total_energy += penalty
        
        return total_energy

    def compute_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the Γ-convergence energy functional.
        
        Args:
            x: Optimization variable (flattened n_partitions × N)
            
        Returns:
            Gradient vector (same shape as x)
        """
        N = len(self.v)
        n = self.n_partitions
        
        # Reshape x to matrix form: (N, n_partitions)
        phi = x.reshape(N, n)
        
        # Initialize gradient
        grad = np.zeros_like(x)
        grad_matrix = grad.reshape(N, n)
        
        # Compute gradient for each partition
        for i in range(n):
            phi_i = phi[:, i]
            
            # Gradient of gradient term: 2ε K u
            grad_gradient = 2 * self.epsilon * (self.K @ phi_i)
            
            # Gradient of interface term: (2/ε) M v × (1 - 2u) where v = u²(1-u)²
            interface_vec = phi_i**2 * (1 - phi_i)**2
            grad_interface = (2/self.epsilon) * (self.M @ interface_vec) * (1 - 2*phi_i)
            
            # Add to gradient
            grad_matrix[:, i] = grad_gradient + grad_interface
        
        # Add penalty gradient if needed
        if self.lambda_penalty > 0:
            for i in range(n):
                phi_i = phi[:, i]
                mean_val = np.mean(phi_i)
                variance = np.var(phi_i)
                
                # Gradient of penalty term
                penalty_grad = self.lambda_penalty * (-2 * (phi_i - mean_val) / N) / (mean_val * (1 - mean_val) + 1e-8)
                grad_matrix[:, i] += penalty_grad
        
        return grad
    
    def constraint_fun(self, x: np.ndarray) -> np.ndarray:
        """
        Compute constraint functions for PySLSQP.
        
        Constraints:
        1. Partition constraint: Σ_{i=1}^n u_i = 1 (everywhere)
        2. Area constraint: ∫_Ω u_i = A/n (equal areas)
        
        Args:
            x: Flattened vector of partition functions
            
        Returns:
            Constraint values (should be zero at solution)
        """
        N = len(self.v)
        n = self.n_partitions
        phi = x.reshape(N, n)
        
        # Row sum constraints (partition constraint): Σ u_i = 1
        # We enforce this for all but the last row (redundant constraint)
        row_sums = np.sum(phi, axis=1)[:-1] - 1.0
        
        # Area constraints (equal area): ∫ u_i = A/n
        # We enforce this for all but the last partition (redundant constraint)
        area_sums = self.v @ phi
        target_area = self.theoretical_total_area / n
        area_constraints = area_sums[:-1] - target_area
        
        return np.concatenate([row_sums, area_constraints])
    
    def constraint_jac(self, x: np.ndarray) -> np.ndarray:
        """
        Compute analytic Jacobian of constraint functions.
        
        Args:
            x: Flattened vector of partition functions
            
        Returns:
            Constraint Jacobian matrix
        """
        N = len(self.v)
        n = self.n_partitions
        
        # Row sum Jacobian: (N-1) x (N*n)
        # For each row i, the Jacobian is 1 for all entries in that row
        row_sum_jac = np.zeros((N-1, N * n))
        for i in range(N-1):
            row_sum_jac[i, i::N] = 1.0
        
        # Area Jacobian: (n-1) x (N*n)
        # For each partition i, the Jacobian is v for that partition's block
        area_jac = np.zeros((n-1, N * n))
        for i in range(n-1):
            area_jac[i, i*N:(i+1)*N] = self.v
        
        return np.vstack([row_sum_jac, area_jac])
    
    def compute_area_evolution(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the area of each partition at the current point.
        
        Args:
            x: Flattened vector of partition functions
            
        Returns:
            Array of partition areas
        """
        N = len(self.v)
        n_partitions = self.n_partitions
        x_reshaped = x.reshape(N, n_partitions)
        return self.v @ x_reshaped

    @log_performance("optimization")
    def optimize(self, x0: np.ndarray = None, maxiter: int = 100, ftol: float = 1e-8, 
                eps: float = 1e-8, disp: bool = False, use_analytic: bool = True, 
                log_frequency: int = 50, use_last_valid_iterate: bool = True, 
                is_mesh_refinement: bool = False, results_dir: str = None, 
                run_name: str = None, hot_start_file: str = None, 
                save_itr: str = None, initial_condition_method: str = "random",
                validate_initial: bool = True, process_initial: bool = True) -> Tuple[np.ndarray, bool]:
        """
        Optimize using PySLSQP with optional analytic gradients and hot-start support.
        
        Args:
            x0: Initial point (if None, will be generated)
            maxiter: Maximum number of iterations
            ftol: Function tolerance
            eps: Gradient tolerance
            disp: Whether to display optimization progress
            use_analytic: Whether to use analytic gradients
            log_frequency: Frequency of logging (every N iterations)
            use_last_valid_iterate: Whether to return last valid iterate on failure
            is_mesh_refinement: Whether this is a mesh refinement step
            results_dir: Directory to save results
            run_name: Name for this optimization run
            hot_start_file: Path to PySLSQP HDF5 file for hot-start continuation
            save_itr: PySLSQP save_itr parameter ('major' or 'all')
            initial_condition_method: Method to generate initial condition if x0 is None
            validate_initial: Whether to validate the initial condition
            process_initial: Whether to process/clean the initial condition
            
        Returns:
            Tuple of (optimized point, success flag)
        """
        # Handle initial condition
        if x0 is None:
            self.logger.info(f"Generating initial condition using method: {initial_condition_method}")
            x0 = self.generate_initial_condition(method=initial_condition_method)
        
        # Process initial condition if requested
        if process_initial:
            self.logger.info("Processing initial condition...")
            x0 = self.process_initial_condition(x0, normalize=True)
        
        # Validate initial condition if requested
        if validate_initial:
            if not self.validate_initial_condition(x0):
                raise ValueError("Initial condition validation failed")
        
        # Set up results directory and filenames
        if results_dir is None:
            results_dir = "results"
        if run_name is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            run_name = f"ring_optimization_{timestamp}"
            
        os.makedirs(results_dir, exist_ok=True)
        summary_filename = os.path.join(results_dir, f"{run_name}_summary.out")
        internal_data_filename = os.path.join(results_dir, f"{run_name}_internal_data.hdf5")
        
        # Store run information
        self.optimization_run_name = run_name
        self.optimization_results_dir = results_dir
        self.summary_file = summary_filename
        self.internal_data_file = internal_data_filename
        
        # Determine if using hot-start
        using_hot_start = hot_start_file and os.path.exists(hot_start_file)
        
        if not is_mesh_refinement:
            # Initialize logging with initial point
            self.log = {
                'iterations': [0],
                'energy_changes': [0.0],
                'warnings': [],
                'area_evolution': [self.compute_area_evolution(x0)] if not using_hot_start else []
            }
            
            # Log initial state
            if not using_hot_start:
                initial_energy = self.compute_energy(x0)
                self.logger.info(f"Initial state before optimization:")
                self.logger.info(f"  Energy: {initial_energy:.6e}")
                self.logger.info(f"  Initial condition shape: {x0.shape}")
                self.logger.info(f"  Detailed optimization data will be saved to:")
                self.logger.info(f"    Summary: {summary_filename}")
                self.logger.info(f"    Internal data: {internal_data_filename}")
        else:
            # For continuation, initialize empty logs
            self.log = {
                'iterations': [],
                'energy_changes': [],
                'warnings': [],
                'area_evolution': []
            }
        
        # Initialize iterate tracking
        self.prev_x = None
        self.curr_x = None
        self.log_frequency = log_frequency
        self.use_last_valid_iterate = use_last_valid_iterate
        self.use_analytic = use_analytic
        
        # Run optimization with PySLSQP
        try:
            # Set bounds: 0 ≤ x ≤ 1
            problem_size = len(x0)
            xl = np.zeros(problem_size)
            xu = np.ones(problem_size)
            
            # Number of equality constraints
            meq = len(self.v) - 1 + self.n_partitions - 1
            
            # Configure PySLSQP parameters
            pyslsqp_params = {
                'obj': self.compute_energy,
                'grad': self.compute_gradient if use_analytic else None,
                'con': self.constraint_fun,
                'jac': self.constraint_jac,
                'meq': meq,
                'xl': xl,
                'xu': xu,
                'maxiter': maxiter,
                'acc': ftol,
                'iprint': 0 if not disp else 2,
                'callback': None if using_hot_start else self.callback,
                'summary_filename': summary_filename,
                'save_vars': ['x', 'objective', 'constraints', 'gradient', 'jacobian'],
                'save_itr': save_itr or 'major',
                'save_filename': internal_data_filename
            }
            
            # Add hot-start parameter if available
            if using_hot_start:
                pyslsqp_params['hot_start'] = True
                pyslsqp_params['load_filename'] = hot_start_file
                self.logger.info(f"🔥 Hot-start enabled with data from: {hot_start_file}")
            
            if using_hot_start:
                self.logger.info("Starting PySLSQP optimization with hot-start...")
            else:
                self.logger.info(f"Starting PySLSQP optimization with "
                               f"{'analytic' if use_analytic else 'finite-difference'} gradients...")
            
            # Run optimization
            result = pyslsqp.optimize(x0, **pyslsqp_params)
            
            x_opt = result['x']
            success = bool(result['success'])
            
            self.logger.info(f"PySLSQP optimization completed:")
            self.logger.info(f"  Success: {success}")
            self.logger.info(f"  Summary saved to: {summary_filename}")
            self.logger.info(f"  Internal data saved to: {internal_data_filename}")
            
        except Exception as e:
            self.logger.error(f"PySLSQP optimization failed: {e}")
            return x0, False
        
        # Handle unsuccessful termination
        if not success and self.prev_x is not None and self.use_last_valid_iterate:
            self.logger.warning("Returning last valid iterate before unsuccessful termination.")
            # Remove last iteration from logs
            for key in ['iterations', 'energy_changes', 'area_evolution']:
                if self.log[key]:
                    self.log[key].pop()
            return self.prev_x.copy(), success
        else:
            return x_opt, success
    
    def callback(self, xk: np.ndarray):
        """
        Callback function to track optimization progress.
        
        Args:
            xk: Current iterate
        """
        self.prev_x = getattr(self, 'curr_x', None)
        iter_num = len(self.log['iterations'])
        N = len(self.v)
        n = self.n_partitions
        phi = xk.reshape(N, n)
        self.curr_x = xk.copy()
        
        # Track iteration and energy change
        current_energy = self.compute_energy(xk)
        self.log['iterations'].append(iter_num)
        
        if iter_num > 0:
            energy_change = current_energy - self.log.get('last_energy', current_energy)
            self.log['energy_changes'].append(energy_change)
        else:
            self.log['energy_changes'].append(0.0)
        
        # Store current energy for next iteration
        self.log['last_energy'] = current_energy
        
        # Hybrid refinement trigger logic
        patience = self.refine_patience
        delta_energy = self.refine_delta_energy
        grad_tol = self.refine_grad_tol
        constraint_tol = self.refine_constraint_tol
        
        if len(self.log['energy_changes']) >= patience:
            # Check if energy has stabilized
            recent_energy_changes = self.log['energy_changes'][-patience:]
            energy_stable = all(abs(change) < delta_energy for change in recent_energy_changes)
            
            if energy_stable:
                # Check gradient and constraint norms
                grad_norm = np.linalg.norm(self.compute_gradient(xk))
                constraint_violation = np.max(np.abs(self.constraint_fun(xk)))
                
                if grad_norm < grad_tol and constraint_violation < constraint_tol:
                    self.logger.info(f"Refinement triggered at iteration {iter_num} by convergence criteria.")
                    self.logger.info(f"  Energy stable: {energy_stable}")
                    self.logger.info(f"  Gradient norm: {grad_norm:.6e} < {grad_tol}")
                    self.logger.info(f"  Constraint violation: {constraint_violation:.6e} < {constraint_tol}")
                    raise RefinementTriggered()
        
        # Detailed progress every log_frequency iterations
        if iter_num % self.log_frequency == 0:
            self.logger.info(f"  Iteration {iter_num}:")
            self.logger.info(f"    Energy: {current_energy:.6e}")
            if iter_num > 0:
                self.logger.info(f"    Energy change: {self.log['energy_changes'][-1]:.6e}")
        
        # Store area values for each partition
        area_sums = self.v @ phi
        self.log['area_evolution'].append(area_sums.copy())
        
        # Log theoretical target area for reference
        if iter_num % self.log_frequency == 0:
            target_area = self.theoretical_total_area / n
            self.logger.info(f"    Theoretical target area per partition: {target_area:.6e}")
            self.logger.info(f"    Current partition areas: {area_sums}")

    def print_optimization_log(self):
        """Print a summary of the optimization log."""
        self.logger.info("Optimization Log Summary:")
        self.logger.info("=" * 80)
        self.logger.info("Note: Detailed optimization data (energies, gradients, constraints, steps) "
                        "is available in summary and HDF5 files.")
        self.logger.info(f"Total iterations: {len(self.log['iterations'])}")
        self.logger.info(f"Area evolution snapshots: {len(self.log['area_evolution'])}")
        if self.log['warnings']:
            self.logger.info(f"Warnings: {len(self.log['warnings'])}")
            for warning in self.log['warnings']:
                self.logger.info(f"  - {warning}")
        self.logger.info("=" * 80) 