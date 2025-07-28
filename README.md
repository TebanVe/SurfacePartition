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

1. **Prerequisites**: 
    - Install `pyenv` to manage Python versions
    - Install the `pyenv-virtualenv` plugin
    - This project uses Python 3.9.7 as specified in the `.python-version` file

2. **Set up Python Environment**: 
    ```bash
    # Install the required Python version if you don't have it
    pyenv install 3.9.7
    
    # Create a virtualenv named 'ring_partition' using Python 3.9.7
    pyenv virtualenv 3.9.7 ring_partition
    
    # Navigate to the project directory
    cd /path/to/project
    
    # The .python-version file will automatically activate the 'ring_partition' environment
    ```

3. **Install Dependencies**: 
    ```bash
    # With the 'ring_partition' environment active, install dependencies
    pip install -r requirements.txt
    ```

## Usage

### Basic Usage

Run the main optimization script:

```bash
python examples/find_optimal_partition.py
```

### Using Custom Parameters

Create a YAML file with your parameters and use:

```bash
python examples/find_optimal_partition.py --input your_parameters.yaml
```

Example `parameters/input.yaml`:
```yaml
n_partitions: 3
n_radial: 8
n_angular: 16
r_inner: 0.5
r_outer: 1.0
lambda_penalty: 0.01
max_iter: 15000
tol: 1e-6
epsilon: 0.1
seed: 42
```

## Project Structure

- `src/`: Core implementation
  - `ring_mesh.py`: Ring mesh generation and matrix computation
  - `pyslsqp_optimizer.py`: PySLSQP optimization implementation
  - `config.py`: Configuration parameters
  - `projection_iterative.py`: Orthogonal projection for constraints
  - `find_contours.py`: Contour extraction and analysis
  - `plot_utils.py`: Visualization utilities
- `examples/`: Example scripts
  - `find_optimal_partition.py`: Main optimization script
  - `ring_visualization.py`: Ring visualization
- `parameters/`: Configuration files
- `results/`: Output directory
- `visualizations/`: Visualization outputs

## Results

The optimization results will be saved in the `results/` directory, including:
- Solution files (.h5 format)
- Optimization logs
- Visualization images

## References

- Bogosel, B., & Oudet, É. (Year). Partitions of Minimal Length on Manifolds. [Paper reference] 