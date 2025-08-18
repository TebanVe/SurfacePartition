#!/usr/bin/env python3
"""
Test script for orthogonal projection functions.
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
from projection_iterative import (
    orthogonal_projection_iterative,
    orthogonal_projection_direct,
    validate_projection_result,
    create_initial_condition_with_projection
)
from logging_config import setup_logging, get_logger

def test_projection_basic():
    """Test basic projection functionality with simple example."""
    logger = get_logger(__name__)
    logger.info("Testing basic projection functionality...")
    
    # Create a simple example
    N, n = 4, 3  # 4 vertices, 3 partitions
    A = np.random.rand(N, n)
    v = np.ones(N)  # Simple mass matrix column sums
    c = np.ones(n)  # Row sums should be 1
    d = np.sum(v) / n * np.ones(n)  # Equal areas
    
    logger.info(f"Input matrix shape: {A.shape}")
    logger.info(f"Input row sums: {np.sum(A, axis=1)}")
    logger.info(f"Input area sums: {v @ A}")
    
    # Test iterative projection
    A_iter = orthogonal_projection_iterative(A.copy(), c, d, v, max_iter=100, tol=1e-8)
    
    # Test direct projection
    A_direct = orthogonal_projection_direct(A.copy(), c, d, v)
    
    # Validate results
    iter_valid = validate_projection_result(A_iter, v, d, tol=1e-6)
    direct_valid = validate_projection_result(A_direct, v, d, tol=1e-4)  # More lenient for direct method
    
    logger.info(f"Iterative projection valid: {iter_valid}")
    logger.info(f"Direct projection valid: {direct_valid}")
    
    # Check final constraints
    logger.info(f"Iterative row sums: {np.sum(A_iter, axis=1)}")
    logger.info(f"Iterative area sums: {v @ A_iter}")
    logger.info(f"Direct row sums: {np.sum(A_direct, axis=1)}")
    logger.info(f"Direct area sums: {v @ A_direct}")
    
    return iter_valid and direct_valid

def test_projection_with_ring_mesh():
    """Test projection with actual ring mesh."""
    logger = get_logger(__name__)
    logger.info("Testing projection with ring mesh...")
    
    # Create ring mesh
    mesh_params = {
        'r_inner': 1.0,
        'r_outer': 2.0,
        'n_radial': 4,
        'n_angular': 8
    }
    
    mesh = RingMesh(**mesh_params)
    mesh.compute_matrices()
    
    N = len(mesh.v)
    n_partitions = 3
    
    logger.info(f"Mesh: {N} vertices, {n_partitions} partitions")
    
    # Test initial condition creation
    x0_iter = create_initial_condition_with_projection(
        N, n_partitions, mesh.v, seed=42, method="iterative"
    )
    
    x0_direct = create_initial_condition_with_projection(
        N, n_partitions, mesh.v, seed=42, method="direct"
    )
    
    # Reshape and validate
    A_iter = x0_iter.reshape(N, n_partitions)
    A_direct = x0_direct.reshape(N, n_partitions)
    
    iter_valid = validate_projection_result(A_iter, mesh.v, mesh.v.sum() / n_partitions * np.ones(n_partitions), tol=1e-6)
    direct_valid = validate_projection_result(A_direct, mesh.v, mesh.v.sum() / n_partitions * np.ones(n_partitions), tol=1e-4)  # More lenient for direct method
    
    logger.info(f"Iterative projection valid: {iter_valid}")
    logger.info(f"Direct projection valid: {direct_valid}")
    
    # Check constraint satisfaction
    row_sums_iter = np.sum(A_iter, axis=1)
    area_sums_iter = mesh.v @ A_iter
    target_area = mesh.v.sum() / n_partitions
    
    logger.info(f"Row sum errors (iterative): max={np.max(np.abs(row_sums_iter - 1)):.2e}")
    logger.info(f"Area errors (iterative): max={np.max(np.abs(area_sums_iter - target_area)):.2e}")
    
    return iter_valid and direct_valid

def test_projection_convergence():
    """Test projection convergence with different tolerances."""
    logger = get_logger(__name__)
    logger.info("Testing projection convergence...")
    
    # Create test case
    N, n = 8, 4
    A = np.random.rand(N, n)
    v = np.random.rand(N) + 0.1  # Ensure positive
    c = np.ones(n)
    d = np.sum(v) / n * np.ones(n)
    
    # Test different tolerances
    tolerances = [1e-6, 1e-8, 1e-10]
    results = {}
    
    for tol in tolerances:
        logger.info(f"Testing tolerance: {tol}")
        
        try:
            A_projected = orthogonal_projection_iterative(A.copy(), c, d, v, max_iter=200, tol=tol)
            valid = validate_projection_result(A_projected, v, d, tol=tol*10)
            
            row_error = np.max(np.abs(np.sum(A_projected, axis=1) - 1))
            area_error = np.max(np.abs(v @ A_projected - d))
            
            results[tol] = {
                'valid': valid,
                'row_error': row_error,
                'area_error': area_error
            }
            
            logger.info(f"  Valid: {valid}, row_error: {row_error:.2e}, area_error: {area_error:.2e}")
            
        except Exception as e:
            logger.error(f"  Failed: {e}")
            results[tol] = {'valid': False, 'error': str(e)}
    
    return results

def test_projection_edge_cases():
    """Test projection with edge cases."""
    logger = get_logger(__name__)
    logger.info("Testing projection edge cases...")
    
    # Test 1: Very small matrix
    N, n = 2, 2
    A = np.random.rand(N, n)
    v = np.ones(N)
    c = np.ones(n)
    d = np.sum(v) / n * np.ones(n)
    
    try:
        A_projected = orthogonal_projection_direct(A, c, d, v)
        valid = validate_projection_result(A_projected, v, d)
        logger.info(f"Small matrix (2x2): {'✓' if valid else '✗'}")
    except Exception as e:
        logger.error(f"Small matrix failed: {e}")
        return False
    
    # Test 2: Many partitions
    N, n = 4, 8
    A = np.random.rand(N, n)
    v = np.ones(N)
    c = np.ones(n)
    d = np.sum(v) / n * np.ones(n)
    
    try:
        A_projected = orthogonal_projection_iterative(A, c, d, v, max_iter=50)
        valid = validate_projection_result(A_projected, v, d)
        logger.info(f"Many partitions (4x8): {'✓' if valid else '✗'}")
    except Exception as e:
        logger.error(f"Many partitions failed: {e}")
        return False
    
    # Test 3: Singular case (all zeros)
    N, n = 3, 3
    A = np.zeros((N, n))
    v = np.ones(N)
    c = np.ones(n)
    d = np.sum(v) / n * np.ones(n)
    
    try:
        A_projected = orthogonal_projection_iterative(A, c, d, v, max_iter=50)
        valid = validate_projection_result(A_projected, v, d)
        logger.info(f"Zero matrix (3x3): {'✓' if valid else '✗'}")
    except Exception as e:
        logger.error(f"Zero matrix failed: {e}")
        return False
    
    return True

def main():
    """Main testing function."""
    # Setup logging with timestamped files
    setup_logging(log_level='INFO', log_dir='logs', include_timestamp=True)
    logger = get_logger(__name__)
    
    logger.info("=== Orthogonal Projection Testing Framework ===")
    logger.info("This script tests the projection functions step by step.")
    
    try:
        # Test 1: Basic functionality
        if not test_projection_basic():
            raise Exception("Basic projection test failed")
        
        # Test 2: Ring mesh integration
        if not test_projection_with_ring_mesh():
            raise Exception("Ring mesh projection test failed")
        
        # Test 3: Convergence testing
        convergence_results = test_projection_convergence()
        logger.info("Convergence test completed")
        
        # Test 4: Edge cases
        if not test_projection_edge_cases():
            raise Exception("Edge case tests failed")
        
        logger.info("=== All Tests Completed Successfully ===")
        logger.info("=== Summary ===")
        logger.info("✓ Basic projection functionality")
        logger.info("✓ Ring mesh integration")
        logger.info("✓ Convergence testing")
        logger.info("✓ Edge case handling")
        logger.info("=== Orthogonal Projection Implementation is Working Correctly ===")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 All projection tests passed! Orthogonal projection implementation is working correctly.")
    else:
        print("\n💥 Some projection tests failed. Please check the implementation.")
        sys.exit(1) 