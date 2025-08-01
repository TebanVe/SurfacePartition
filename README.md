# Ring Partition Project

This project implements the optimization method described in "Partitions of Minimal Length on Manifolds" by Bogosel and Oudet for partitioning a ring (annulus) in R² into equal area regions while minimizing the total perimeter.

## Mathematical Framework

The project uses a Γ-convergence approach with the energy functional:

$$J_ε(u) = ε \int_R |∇u|^2 + \frac{1}{ε} \int_R u^2(1 - u)^2$$

For partitions, we minimize:

$$\sum_{i=1}^n J_ε(u_i)$$

Subject to constraints:
- Partition constraint: $\sum_{i=1}^n u_i = 1$
- Area constraint: $\int_R u_i = A/n$ for each partition

Where $R$ is the ring domain in R².

## Installation

### Prerequisites

**Important**: This project requires Python 3.9.7 specifically due to PySLSQP compatibility issues. Python 3.13+ causes compilation errors with PySLSQP.

1. **Install pyenv and pyenv-virtualenv**:
   ```bash
   # macOS
   brew install pyenv pyenv-virtualenv
   
   # Linux
   curl https://pyenv.run | bash
   ```

2. **Set up Python Environment**:
   ```bash
   # Install Python 3.9.7
   pyenv install 3.9.7
   
   # Create virtual environment
   pyenv virtualenv 3.9.7 ringtest-3.9
   
   # Navigate to project directory
   cd /path/to/RingTest
   
   # Activate environment (automatic via .python-version)
   pyenv local ringtest-3.9
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Troubleshooting

**PySLSQP Installation Issues**: If you encounter compilation errors like `library 'ifcore' not found`, ensure you're using Python 3.9.7. The project includes a `.python-version` file that should automatically activate the correct environment.

## Current Implementation Status

### ✅ Implemented Features

- **Ring Mesh Generation**: 2D annulus mesh with triangular elements
- **FEM Matrix Computation**: Mass and stiffness matrices for Γ-convergence
- **PySLSQP Optimizer**: Complete optimization implementation with:
  - Energy functional computation
  - Analytic gradients and Jacobians
  - Constraint handling (partition + area)
  - Initial condition generation and validation
  - Enhanced logging and performance monitoring
- **Testing Framework**: Comprehensive test suite for all components
- **Logging System**: Timestamped logs with performance tracking

### 🔄 In Progress

- **Projection Algorithm**: Orthogonal projection for constraint satisfaction
- **Main Optimization Script**: Full optimization pipeline
- **Visualization Tools**: Results plotting and analysis

## Usage

### Testing the Implementation

```bash
# Test mesh and matrix computation
python examples/test_mesh_matrices.py

# Test optimizer functionality
python examples/test_optimizer.py

# Run main program (basic structure)
python examples/find_optimal_partition.py
```

### Configuration

The project uses configuration files in `parameters/` directory. See `parameters/input.yaml` for available parameters.

## Project Structure

```
RingTest/
├── src/                          # Core implementation
│   ├── ring_mesh.py             # Ring mesh generation
│   ├── pyslsqp_optimizer.py     # PySLSQP optimization
│   ├── config.py                # Configuration management
│   ├── plot_utils.py            # Visualization utilities
│   └── logging_config.py        # Logging system
├── examples/                     # Example scripts
│   ├── find_optimal_partition.py # Main program
│   ├── test_mesh_matrices.py    # Mesh testing
│   └── test_optimizer.py        # Optimizer testing
├── parameters/                   # Configuration files
├── logs/                        # Timestamped log files
└── results/                     # Optimization results
```

## Key Features

### Ring Mesh (`src/ring_mesh.py`)
- 2D annulus mesh generation in polar coordinates
- FEM matrix computation (mass and stiffness)
- Property-based access (`mesh.K`, `mesh.M`, `mesh.v`)
- Mesh quality validation and statistics

### PySLSQP Optimizer (`src/pyslsqp_optimizer.py`)
- Γ-convergence energy functional implementation
- Analytic gradients and constraint Jacobians
- Initial condition generation and validation
- Performance monitoring with decorators
- Comprehensive logging and error handling

### Logging System (`src/logging_config.py`)
- Timestamped log files
- Performance monitoring decorators
- Component-specific logging
- Automatic log rotation

## Results

Optimization results are saved in the `results/` directory with timestamped filenames. Logs are stored in `logs/` with detailed performance metrics.

## References

- Bogosel, B., & Oudet, É. (Year). Partitions of Minimal Length on Manifolds. [Paper reference]
- PySLSQP: [https://github.com/danielzuegner/pyslsqp](https://github.com/danielzuegner/pyslsqp) 