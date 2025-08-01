#!/usr/bin/env python3
"""
Test script for PySLSQPOptimizer functionality.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import after adding to path
from ring_mesh import RingMesh
from config import Config
from pyslsqp_optimizer import PySLSQPOptimizer, RefinementTriggered
from logging_config import setup_logging, get_logger

def test_optimizer_initialization():
    """Test optimizer initialization with valid parameters."""
    logger = get_logger(__name__)
    logger.info("Testing optimizer initialization...")
    
    # Create a simple mesh
    mesh_params = {
        'r_inner': 1.0,
        'r_outer': 2.0,
        'n_radial': 4,
        'n_angular': 8
    }
    
    mesh = RingMesh(**mesh_params)
    mesh.compute_matrices()
    
    # Test optimizer initialization
    optimizer_params = {
        'K': mesh.K,
        'M': mesh.M,
        'v': mesh.v,
        'n_partitions': 3,
        'epsilon': 0.1,
        'r_inner': 1.0,
        'r_outer': 2.0,
        'lambda_penalty': 1.0
    }
    
    optimizer = PySLSQPOptimizer(**optimizer_params)
    
    logger.info("✓ Optimizer initialization successful")
    logger.info(f"  Theoretical area: {optimizer.theoretical_total_area:.6f}")
    logger.info(f"  Discretized area: {optimizer.total_area:.6f}")
    logger.info(f"  Target std dev: {optimizer.starget:.6f}")
    
    return optimizer, mesh

def test_energy_computation(optimizer):
    """Test energy computation with random initial point."""
    logger = get_logger(__name__)
    logger.info("Testing energy computation...")
    
    # Test initial condition generation
    logger.info("Testing initial condition generation...")
    
    # Test different generation methods
    methods = ["random", "uniform", "radial"]
    valid_x0 = None
    for method in methods:
        x0 = optimizer.generate_initial_condition(method=method)
        logger.info(f"✓ Generated initial condition with method '{method}': shape = {x0.shape}")
        
        # Validate the generated initial condition
        if optimizer.validate_initial_condition(x0):
            logger.info(f"✓ Initial condition from '{method}' method is valid")
            valid_x0 = x0  # Store the first valid one
        else:
            logger.error(f"✗ Initial condition from '{method}' method is invalid")
            # Try to process it
            try:
                x0_processed = optimizer.process_initial_condition(x0, normalize=True)
                if optimizer.validate_initial_condition(x0_processed):
                    logger.info(f"✓ Initial condition from '{method}' method is valid after processing")
                    valid_x0 = x0_processed
                else:
                    logger.error(f"✗ Initial condition from '{method}' method is still invalid after processing")
            except Exception as e:
                logger.error(f"✗ Failed to process initial condition from '{method}' method: {e}")
    
    if valid_x0 is None:
        logger.error("✗ No valid initial condition could be generated")
        return None
    
    # Use the first valid initial condition for further testing
    x0 = valid_x0
    
    # Test energy computation
    energy = optimizer.compute_energy(x0)
    logger.info(f"✓ Energy computation successful: {energy:.6e}")
    
    # Test gradient computation
    gradient = optimizer.compute_gradient(x0)
    logger.info(f"✓ Gradient computation successful: norm = {np.linalg.norm(gradient):.6e}")
    
    # Test constraint computation
    constraints = optimizer.constraint_fun(x0)
    constraint_violation = np.max(np.abs(constraints))
    logger.info(f"✓ Constraint computation successful: max violation = {constraint_violation:.6e}")
    
    return x0

def test_constraint_jacobian(optimizer, x0):
    """Test constraint Jacobian computation."""
    logger = get_logger(__name__)
    logger.info("Testing constraint Jacobian...")
    
    # Compute Jacobian
    jac = optimizer.constraint_jac(x0)
    
    # Test dimensions
    N = len(optimizer.v)
    n = optimizer.n_partitions
    expected_rows = (N - 1) + (n - 1)  # row constraints + area constraints
    expected_cols = N * n
    
    if jac.shape == (expected_rows, expected_cols):
        logger.info(f"✓ Jacobian dimensions correct: {jac.shape}")
    else:
        logger.error(f"✗ Jacobian dimensions incorrect: got {jac.shape}, expected ({expected_rows}, {expected_cols})")
        return False
    
    # Test that Jacobian is not all zeros
    if np.any(jac != 0):
        logger.info("✓ Jacobian contains non-zero elements")
    else:
        logger.error("✗ Jacobian is all zeros")
        return False
    
    return True

def test_area_evolution(optimizer, x0):
    """Test area evolution computation."""
    logger = get_logger(__name__)
    logger.info("Testing area evolution...")
    
    areas = optimizer.compute_area_evolution(x0)
    
    if len(areas) == optimizer.n_partitions:
        logger.info(f"✓ Area evolution successful: {areas}")
    else:
        logger.error(f"✗ Area evolution failed: got {len(areas)} areas, expected {optimizer.n_partitions}")
        return False
    
    return True

def test_optimization_interface(optimizer, x0):
    """Test the optimization interface (without actually running optimization)."""
    logger = get_logger(__name__)
    logger.info("Testing optimization interface...")
    
    try:
        # Test 1: Optimization with provided initial condition
        result, success = optimizer.optimize(
            x0, 
            maxiter=5,  # Very small for testing
            ftol=1e-6,
            disp=False,
            use_analytic=True,
            log_frequency=10,
            validate_initial=True,
            process_initial=True
        )
        
        logger.info(f"✓ Optimization with provided initial condition completed")
        logger.info(f"  Success: {success}")
        logger.info(f"  Result shape: {result.shape}")
        
        # Test 2: Optimization with auto-generated initial condition
        result2, success2 = optimizer.optimize(
            x0=None,  # Let optimizer generate initial condition
            maxiter=5,  # Very small for testing
            ftol=1e-6,
            disp=False,
            use_analytic=True,
            log_frequency=10,
            initial_condition_method="uniform",
            validate_initial=True,
            process_initial=True
        )
        
        logger.info(f"✓ Optimization with auto-generated initial condition completed")
        logger.info(f"  Success: {success2}")
        logger.info(f"  Result shape: {result2.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Optimization interface test failed: {e}")
        return False

def test_parameter_validation():
    """Test parameter validation."""
    logger = get_logger(__name__)
    logger.info("Testing parameter validation...")
    
    # Create a simple mesh for testing
    mesh_params = {
        'r_inner': 1.0,
        'r_outer': 2.0,
        'n_radial': 3,
        'n_angular': 4
    }
    
    mesh = RingMesh(**mesh_params)
    mesh.compute_matrices()
    
    # Test valid parameters
    try:
        optimizer = PySLSQPOptimizer(
            K=mesh.K,
            M=mesh.M,
            v=mesh.v,
            n_partitions=2,
            epsilon=0.1,
            r_inner=1.0,
            r_outer=2.0
        )
        logger.info("✓ Valid parameters accepted")
    except Exception as e:
        logger.error(f"✗ Valid parameters rejected: {e}")
        return False
    
    # Test invalid parameters
    invalid_tests = [
        ("n_partitions < 2", {'n_partitions': 1}),
        ("epsilon <= 0", {'epsilon': 0.0}),
        ("r_inner <= 0", {'r_inner': 0.0}),
        ("r_outer <= 0", {'r_outer': 0.0}),
        ("r_inner >= r_outer", {'r_inner': 2.0, 'r_outer': 1.0}),
    ]
    
    for test_name, invalid_params in invalid_tests:
        try:
            # Create base parameters
            base_params = {
                'K': mesh.K,
                'M': mesh.M,
                'v': mesh.v,
                'n_partitions': 2,
                'epsilon': 0.1,
                'r_inner': 1.0,
                'r_outer': 2.0
            }
            # Update with invalid parameters
            base_params.update(invalid_params)
            
            optimizer = PySLSQPOptimizer(**base_params)
            logger.error(f"✗ {test_name} should have been rejected")
            return False
        except ValueError:
            logger.info(f"✓ {test_name} correctly rejected")
    
    return True

def test_initial_condition_handling(optimizer):
    """Test initial condition validation and processing."""
    logger = get_logger(__name__)
    logger.info("Testing initial condition handling...")
    
    N = len(optimizer.v)
    n = optimizer.n_partitions
    
    # Test 1: Valid initial condition
    x0_valid = optimizer.generate_initial_condition("uniform")
    if optimizer.validate_initial_condition(x0_valid):
        logger.info("✓ Valid initial condition correctly validated")
    else:
        logger.error("✗ Valid initial condition incorrectly rejected")
        return False
    
    # Test 2: Invalid initial condition (wrong dimensions)
    x0_invalid_dim = np.random.rand(N * n + 1)
    if not optimizer.validate_initial_condition(x0_invalid_dim):
        logger.info("✓ Invalid dimension correctly rejected")
    else:
        logger.error("✗ Invalid dimension incorrectly accepted")
        return False
    
    # Test 3: Invalid initial condition (out of bounds)
    x0_invalid_bounds = np.random.rand(N * n) * 2  # Values > 1
    if not optimizer.validate_initial_condition(x0_invalid_bounds):
        logger.info("✓ Invalid bounds correctly rejected")
    else:
        logger.error("✗ Invalid bounds incorrectly accepted")
        return False
    
    # Test 4: Invalid initial condition (partition constraint violated)
    x0_invalid_partition = np.random.rand(N * n)
    # Don't normalize, so partition constraint is violated
    if not optimizer.validate_initial_condition(x0_invalid_partition):
        logger.info("✓ Invalid partition constraint correctly rejected")
    else:
        logger.error("✗ Invalid partition constraint incorrectly accepted")
        return False
    
    # Test 5: Initial condition processing
    x0_raw = np.random.rand(N * n) * 2  # Raw data with violations
    x0_processed = optimizer.process_initial_condition(x0_raw, normalize=True)
    
    if optimizer.validate_initial_condition(x0_processed):
        logger.info("✓ Initial condition processing successful")
    else:
        logger.error("✗ Initial condition processing failed")
        return False
    
    return True

def main():
    """Main testing function."""
    # Setup logging with timestamped files
    setup_logging(log_level='INFO', log_dir='logs', include_timestamp=True)
    logger = get_logger(__name__)
    
    logger.info("=== PySLSQPOptimizer Testing Framework ===")
    logger.info("This script tests the ring optimizer implementation step by step.")
    
    try:
        # Test 1: Initialization
        optimizer, mesh = test_optimizer_initialization()
        
        # Test 2: Energy and gradient computation
        x0 = test_energy_computation(optimizer)
        
        # Test 3: Constraint Jacobian
        if not test_constraint_jacobian(optimizer, x0):
            raise Exception("Constraint Jacobian test failed")
        
        # Test 4: Area evolution
        if not test_area_evolution(optimizer, x0):
            raise Exception("Area evolution test failed")
        
        # Test 5: Parameter validation
        if not test_parameter_validation():
            raise Exception("Parameter validation test failed")
        
        # Test 6: Initial condition handling
        if not test_initial_condition_handling(optimizer):
            raise Exception("Initial condition handling test failed")
        
        # Test 7: Optimization interface (brief test)
        if not test_optimization_interface(optimizer, x0):
            raise Exception("Optimization interface test failed")
        
        logger.info("=== All Tests Completed Successfully ===")
        logger.info("=== Summary ===")
        logger.info("✓ Optimizer initialization")
        logger.info("✓ Energy and gradient computation")
        logger.info("✓ Constraint functions and Jacobian")
        logger.info("✓ Area evolution tracking")
        logger.info("✓ Parameter validation")
        logger.info("✓ Initial condition handling")
        logger.info("✓ Optimization interface")
        logger.info("=== Ring Optimizer Implementation is Working Correctly ===")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 All optimizer tests passed! Ring optimizer implementation is working correctly.")
    else:
        print("\n💥 Some optimizer tests failed. Please check the implementation.")
        sys.exit(1) 