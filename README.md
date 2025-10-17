# Surface Partition Optimization Framework

This project implements a surface-agnostic framework for finding minimal-perimeter partitions on triangulated surfaces, based on the optimization method described in "Partitions of Minimal Length on Manifolds" by Bogosel and Oudet. The framework supports any triangulated surface (2D or 3D) and provides both PySLSQP and Projected Gradient Descent optimizers with mesh refinement capabilities.

## Mathematical Framework

The project uses a Γ-convergence approach with the energy functional:

$$J_ε(u) = ε \int_S |∇_τ u|^2 + \frac{1}{ε} \int_S u^2(1 - u)^2$$

For partitions, we minimize:

$$\sum_{i=1}^n J_ε(u_i)$$

Subject to constraints:
- **Partition constraint**: $\sum_{i=1}^n u_i = 1$ at each vertex
- **Area constraint**: $\int_S u_i = A/n$ for each partition

Where $S$ is any triangulated surface and $∇_τ$ denotes the tangential gradient.

## Key Features

### 🎯 **Perimeter Refinement** (NEW)
- **Section 5 Implementation**: Direct perimeter minimization on extracted contours
- **Constrained Optimization**: Equal-area constraints with analytical gradients
- **Steiner Trees**: Optimal handling of triple points (3 regions meet)
- **Topology Management**: Variable points on mesh edges with automatic detection
- **Accurate Perimeter Values**: Match paper benchmarks (Tables 1 & 2)
- See [docs/PERIMETER_REFINEMENT.md](docs/PERIMETER_REFINEMENT.md) for details

### 🚀 **Surface-Agnostic Design**
- **TriMesh**: Universal triangle mesh class supporting both 2D and 3D surfaces
- **Surface Providers**: Modular system for different surface types (ring, sphere, torus, etc.)
- **P1 FEM Assembly**: Automatic mass and stiffness matrix computation for any triangulation

### 🔧 **Dual Optimizer Support**
- **PySLSQP**: Sequential least squares programming optimizer for constrained problems
- **Projected Gradient Descent (PGD)**: Custom gradient descent with constraint projection
- **Configurable Parameters**: Extensive configuration options for both optimizers

### 📈 **Mesh Refinement System**
- **Multi-level Optimization**: Progressive mesh refinement with solution interpolation
- **Automatic Refinement**: Smart triggers based on convergence metrics and plateaus
- **Memory Efficient**: Optimized for handling large meshes across refinement levels

### 📊 **Enhanced Analysis Tools**
- **Optimization Analyzer**: Comprehensive result analysis with constraint evolution plots
- **Island Analysis**: Tools for diagnosing partition quality and identifying ambiguous regions
- **Visualization Suite**: Advanced plotting and mesh visualization capabilities

### 💾 **Robust Data Management**
- **HDF5 Output**: Efficient storage of optimization iterates and solutions
- **Configurable Logging**: Flexible logging with performance monitoring
- **Metadata Tracking**: Comprehensive run metadata and configuration persistence

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

## Project Structure

```
RingTest/
├── src/                          # Core implementation
│   ├── core/                     # Surface-agnostic core components
│   │   ├── tri_mesh.py          # Universal triangle mesh class
│   │   ├── pyslsqp_optimizer.py # PySLSQP optimization engine
│   │   ├── pgd_optimizer.py     # Projected gradient descent optimizer
│   │   └── interpolation.py     # Solution interpolation utilities
│   ├── surfaces/                 # Surface-specific providers
│   │   ├── ring.py              # Ring/annulus surface provider
│   │   └── torus.py             # Torus of revolution (R3) provider
│   ├── config.py                 # Configuration management
│   ├── plot_utils.py             # 2D visualization utilities (Matplotlib)
│   ├── plot_utils_3d.py          # 3D visualization utilities (PyVista)
│   ├── logging_config.py         # Logging system
│   ├── projection_iterative.py   # Constraint projection algorithms
│   ├── find_contours.py          # Contour extraction utilities (R2/R3)
│   └── island_analysis.py        # Partition quality analysis
├── examples/                      # Example scripts and analysis tools
│   ├── find_surface_partition.py # Main optimization orchestrator
│   ├── optimization_analyzer.py  # Surface-agnostic result analysis
│   ├── surface_visualization.py  # Unified 2D/3D visualization entrypoint
│   ├── island_analyzer.py        # Island detection and analysis
│   ├── test_mesh_matrices.py     # Mesh and matrix testing
│   ├── test_optimizer.py         # Optimizer testing
│   ├── test_projection.py        # Projection algorithm testing
│   └── ring_visualization.py     # Ring-only visualization
├── parameters/                    # Configuration files
│   └── input.yaml                # Default input parameters
├── scripts/                       # Utility scripts
│   └── submit.sh                 # SLURM cluster submission script
├── logs/                         # Timestamped log files
└── results/                      # Optimization results (gitignored)
```

## Usage

### Basic Surface Partition Optimization

```bash
# Run optimization with default parameters
python examples/find_surface_partition.py

# Run with custom input file
python examples/find_surface_partition.py --input parameters/input.yaml

# Specify output directory for solutions
python examples/find_surface_partition.py --input parameters/input.yaml --solution-dir results/my_solutions

# Use specific surface provider ('ring' or 'torus')
python examples/find_surface_partition.py --input parameters/input.yaml --surface ring
python examples/find_surface_partition.py --input parameters/input.yaml --surface torus
```

### Perimeter Refinement (NEW)

After obtaining a relaxed solution, refine the contours to get accurate perimeter values:

```bash
# Step 1: Run Γ-convergence optimization (Section 3)
python examples/find_surface_partition.py --input parameters/input.yaml

# Step 2: Refine contours (Section 5)
python examples/refine_perimeter.py \
    --solution results/run_xyz/solution_level0.h5 \
    --max-iterations 10 \
    --tolerance 1e-7

# Step 3: Visualize refined contours
python examples/surface_visualization.py \
    --solution results/run_xyz/solution_level0.h5 \
    --refined
```

For complete documentation, see [docs/PERIMETER_REFINEMENT.md](docs/PERIMETER_REFINEMENT.md)

### Optimizer Selection

The optimizer is selected in the configuration file (`parameters/input.yaml`):
```yaml
optimizer_type: 'pyslsqp'  # or 'pgd'
```

### Mesh Refinement

Refinement is configured in the YAML file:
```yaml
refinement_levels: 3
n_radial_increment: 2
n_angular_increment: 2
```

### Analysis and Visualization

```bash
# Analyze optimization results
python examples/optimization_analyzer.py --results-dir results/run_20250101_120000_ring_npart2_nr8_na16_lam0.0_seed42

# Analyze multiple runs matching pattern
python examples/optimization_analyzer.py --results-dir results --pattern "npart2_lam0.0"
```

### Testing Components

```bash
# Test mesh generation and matrix computation
python examples/test_mesh_matrices.py

# Test optimizer functionality
python examples/test_optimizer.py

# Test projection algorithms
python examples/test_projection.py
```

### Torus Surface Support

The framework now supports a torus of revolution (R3):
- Provider: `TorusMeshProvider` (`src/surfaces/torus.py`)
- Resolution labels: `nt` (major), `np` (minor)
- Theoretical area: `4π² R r`

Usage:
```bash
# Run optimization on a torus
python examples/find_surface_partition.py --input parameters/input.yaml --surface torus
```

YAML parameters (used when `--surface torus`):
```yaml
n_theta: 32        # samples along major circle
n_phi: 24          # samples along tube circle
R: 1.0             # major radius
r: 0.3             # minor radius
n_theta_increment: 0
n_phi_increment: 0
```

### Surface Visualization (2D/3D)

Use a single script to visualize partitions on any supported surface:

```bash
# 2D (Matplotlib) or 3D (PyVista) depending on solution
python examples/surface_visualization.py --solution <path/to/solution.h5> --level 0.5

# Optional flags
--use-initial          # visualize x0 instead of x_opt
--save out.png         # save image instead of showing a window
--no-fill --no-mesh    # 2D-only styling options
--show-normals         # 3D-only; overlay triangle normals
--normal-scale 0.1     # 3D-only; scale of normals
--color-partition      # 3D-only; light per-face region colors + strong contours
```

Notes:
- 3D rendering (e.g., torus) requires `pyvista` (optional dependency). 2D ring plots do not require it.
- The script automatically infers 2D vs 3D from the solution's vertex dimension.

Flag descriptions (brief):
- `--use-initial`: visualize the initial condition `x0` instead of `x_opt` (auto-saved 3D screenshots add `_initial` suffix).
- `--save <path>`: write the image to file. If omitted, shows an interactive window.
- `--no-fill` / `--no-mesh` (2D): toggle filled partitions or mesh overlay.
- `--show-normals` (3D): overlay triangle normals; `--normal-scale` controls arrow length.
- `--color-partition` (3D): light per-face region colors (approximate) with strong-colored contour lines.

## Configuration

The project uses comprehensive configuration through `parameters/input.yaml`. Key parameters include:

### Surface Configuration
- `n_radial`, `n_angular`: Initial mesh resolution
- `r_inner`, `r_outer`: Ring geometry (for ring surface)
- `refinement_levels`: Number of mesh refinement levels
- `n_radial_increment`, `n_angular_increment`: Resolution increments per level

### Optimization Parameters
- `optimizer_type`: Choose between 'pyslsqp' and 'pgd'
- `n_partitions`: Number of equal-area partitions
- `lambda_penalty`: Penalty parameter for constraint violations
- `epsilon`: Interface width parameter (auto-computed from mesh)

### Refinement Control
- `enable_refinement_triggers`: Enable/disable early refinement
- `refine_gnorm_patience`, `refine_feas_patience`: Plateau detection parameters
- `refine_trigger_mode`: Refinement trigger strategy ('full' or 'energy')

### PGD-Specific Settings
- `h5_save_stride`: HDF5 output frequency
- `h5_save_vars`: Variables to save in HDF5 files
- `run_log_frequency`: Console logging frequency
- `penalty_target_mode`: Constant‑phase penalty target; `fixed` (paper, uses μ_target=1/n) or `adaptive` (uses current μ)
- `penalty_eps`: Small stabilizer added to target denominators

## Core Components

### TriMesh (`src/core/tri_mesh.py`)
- **Universal mesh class** supporting 2D and 3D surfaces
- **P1 FEM assembly** with automatic mass and stiffness matrix computation
- **Memory efficient** sparse matrix operations
- **Mesh statistics** and quality metrics

### Surface Providers (`src/surfaces/`)
- **Modular design** for different surface types
- **Resolution management** with refinement support
- **Metadata generation** for orchestrators and analysis tools

### PySLSQP Optimizer (`src/core/pyslsqp_optimizer.py`)
- **Constrained optimization** with analytic gradients and Jacobians
- **Refinement triggers** based on convergence metrics
- **Comprehensive logging** and performance monitoring

### PGD Optimizer (`src/core/pgd_optimizer.py`)
- **Projected gradient descent** with constraint satisfaction
- **FEM-weighted constant-phase penalty** using `v = 1^T M` and `W = Σ v` for mean/variance, consistent with FEM setting (`docs/starget/constant_phase_penalty_derivation.tex`)
- **Penalty modes**: fixed target (paper) or adaptive target; gradients implemented per weighted formulas
- **Configurable output** for memory efficiency
- **Plateau detection** for intelligent refinement

### Analysis Tools (`examples/optimization_analyzer.py`)
- **Multi-level result analysis** with constraint evolution plots
- **Memory-efficient** handling of large datasets
- **Comprehensive visualization** of optimization metrics

## Advanced Features

### Mesh Refinement System
The framework automatically refines meshes when optimization plateaus, interpolating solutions between levels for improved accuracy.

### Memory Optimization
- **Sparse matrix operations** throughout the pipeline
- **Configurable HDF5 output** to control file sizes
- **Explicit memory management** during multi-level refinement

### Cluster Support
- **SLURM submission script** for UPPMAX (Rackham) cluster
- **Generic design** for other cluster systems
- **Automatic job naming** based on configuration

### Island Analysis
Tools for diagnosing partition quality by identifying ambiguous regions where multiple phases have similar values.

## Results and Output

Optimization runs produce:
- **HDF5 files** with solution iterates and metadata
- **Summary files** with convergence metrics
- **Visualization plots** showing optimization progress
- **Comprehensive logs** with performance metrics
- **Metadata files** for result analysis and reproduction

## References

- Bogosel, B., & Oudet, É. (Year). Partitions of Minimal Length on Manifolds. [Paper reference]
- PySLSQP: [https://github.com/danielzuegner/pyslsqp](https://github.com/danielzuegner/pyslsqp)
- Γ-convergence theory for surface partitioning problems

## Contributing

This project uses a modular, surface-agnostic architecture. To add support for new surfaces:
1. Create a new provider class in `src/surfaces/`
2. Implement the required interface methods
3. Add configuration options as needed
4. Test with the existing analysis tools

## License

[Add your license information here] 