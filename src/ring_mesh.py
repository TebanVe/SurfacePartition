import numpy as np
import scipy.sparse as sparse
import logging

# Handle both relative and absolute imports
try:
    from .logging_config import get_logger, log_performance
except ImportError:
    # Fallback for when running as script
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    from logging_config import get_logger, log_performance

class RingMesh:
    def __init__(self, n_radial, n_angular, r_inner, r_outer):
        """
        Initialize a ring (annulus) mesh.
        
        Parameters:
        -----------
        n_radial : int
            Number of points in the radial direction
        n_angular : int
            Number of points in the angular direction
        r_inner : float
            Inner radius of the ring
        r_outer : float
            Outer radius of the ring
        """
        self.logger = get_logger(__name__)
        
        self.logger.info(f"Creating ring mesh: n_radial={n_radial}, n_angular={n_angular}, "
                        f"r_inner={r_inner}, r_outer={r_outer}")
        
        self.n_radial = n_radial
        self.n_angular = n_angular
        self.r_inner = r_inner
        self.r_outer = r_outer
        
        # Generate vertices
        self.logger.debug("Generating vertices...")
        self.vertices = self._generate_vertices()
        
        # Generate triangles
        self.logger.debug("Generating triangles...")
        self.triangles = self._generate_triangles()
        
        self.faces = self.triangles
        
        # Compute triangle statistics
        self.logger.debug("Computing triangle statistics...")
        self._compute_triangle_statistics()
        
        self.mass_matrix = None
        self.stiffness_matrix = None
        
        self.logger.info(f"Mesh created: {self.get_vertex_count()} vertices, "
                        f"{self.get_triangle_count()} triangles")
        
        # Log theoretical vs computed area
        theoretical_area = np.pi * (self.r_outer**2 - self.r_inner**2)
        self.logger.info(f"Theoretical area: {theoretical_area:.6f}")
        
    def _generate_vertices(self):
        """Generate the vertices of the ring mesh in polar coordinates."""
        self.logger.debug("Generating vertices in polar coordinates...")
        
        r = np.linspace(self.r_inner, self.r_outer, self.n_radial)
        theta = np.linspace(0, 2*np.pi, self.n_angular, endpoint=False)
        
        vertices = []
        for radius in r:
            for angle in theta:
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                vertices.append([x, y])
        
        vertices_array = np.array(vertices)
        self.logger.debug(f"Generated {len(vertices_array)} vertices")
        return vertices_array
    
    def _generate_triangles(self):
        """Generate the triangles of the ring mesh with consistent counterclockwise orientation."""
        self.logger.debug("Generating triangles with counterclockwise orientation...")
        
        triangles = []
        for i in range(self.n_radial - 1):  # Radial direction
            for j in range(self.n_angular):  # Angular direction
                # Get indices of current and next points in both directions
                current = i * self.n_angular + j
                next_radial = (i + 1) * self.n_angular + j
                next_angular = i * self.n_angular + ((j + 1) % self.n_angular)
                next_both = (i + 1) * self.n_angular + ((j + 1) % self.n_angular)
                
                # Add two triangles for each quad, both counterclockwise
                # Triangle 1: [current, next_radial, next_angular] (counterclockwise)
                triangles.append([current, next_radial, next_angular])
                # Triangle 2: [next_radial, next_both, next_angular] (counterclockwise)
                triangles.append([next_radial, next_both, next_angular])
        
        triangles_array = np.array(triangles)
        self.logger.debug(f"Generated {len(triangles_array)} triangles")
        return triangles_array
    
    @log_performance("matrix computation")
    def compute_matrices(self):
        """
        Compute mass and stiffness matrices for 2D ring domain using proper FEM.
        
        For P1 elements on triangles:
        - Mass matrix M[i,j] = ∫_Ω φ_i φ_j dx
        - Stiffness matrix K[i,j] = ∫_Ω ∇φ_i · ∇φ_j dx
        
        where φ_i are the P1 basis functions (linear on each triangle).
        """
        if self.vertices is None or self.triangles is None:
            error_msg = "Mesh must be created before computing matrices"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.info("Starting matrix computation...")
        
        n_vertices = self.vertices.shape[0]
        self.logger.debug(f"Creating matrices of size {n_vertices}x{n_vertices}")
        
        M = sparse.lil_matrix((n_vertices, n_vertices))
        K = sparse.lil_matrix((n_vertices, n_vertices))
        
        def compute_p1_gradients(p1, p2, p3):
            """
            Compute gradients of P1 basis functions on triangle.
            
            For triangle with vertices p1, p2, p3, the P1 basis functions are:
            φ₁(x) = λ₁(x) = [(p2-p3) × (x-p3)] / [(p2-p3) × (p1-p3)]
            φ₂(x) = λ₂(x) = [(p3-p1) × (x-p1)] / [(p3-p1) × (p2-p1)]  
            φ₃(x) = λ₃(x) = [(p1-p2) × (x-p2)] / [(p1-p2) × (p3-p2)]
            
            The gradients are constant on each triangle:
            ∇φ₁ = (p2-p3)⊥ / (2|T|)
            ∇φ₂ = (p3-p1)⊥ / (2|T|)
            ∇φ₃ = (p1-p2)⊥ / (2|T|)
            
            where ⊥ means rotate by 90°: (x,y)⊥ = (-y,x)
            """
            # Compute triangle area
            area = 0.5 * abs(np.cross(p2 - p1, p3 - p1))
            
            # Compute gradients directly in terms of vertex coordinates
            # ∇φ₁ = (p2-p3)⊥ / (2|T|) = (-(y2-y3), x2-x3) / (2|T|)
            grad_phi1 = np.array([-(p2[1] - p3[1]), p2[0] - p3[0]]) / (2 * area)
            
            # ∇φ₂ = (p3-p1)⊥ / (2|T|) = (-(y3-y1), x3-x1) / (2|T|)
            grad_phi2 = np.array([-(p3[1] - p1[1]), p3[0] - p1[0]]) / (2 * area)
            
            # ∇φ₃ = (p1-p2)⊥ / (2|T|) = (-(y1-y2), x1-x2) / (2|T|)
            grad_phi3 = np.array([-(p1[1] - p2[1]), p1[0] - p2[0]]) / (2 * area)
            
            return grad_phi1, grad_phi2, grad_phi3, area
        
        # Assemble matrices element by element
        self.logger.debug("Assembling matrices element by element...")
        for triangle_idx, triangle in enumerate(self.triangles):
            if triangle_idx % 100 == 0:
                self.logger.debug(f"Processing triangle {triangle_idx}/{len(self.triangles)}")
            
            # Get triangle vertices
            p1 = self.vertices[triangle[0]]
            p2 = self.vertices[triangle[1]]
            p3 = self.vertices[triangle[2]]
            
            # Compute gradients and area
            grad1, grad2, grad3, area = compute_p1_gradients(p1, p2, p3)
            
            # Local mass matrix for P1 elements
            # M[i,j] = ∫_T φᵢ φⱼ dx = |T|/12 * (1 + δᵢⱼ) for i≠j, |T|/6 for i=j
            local_mass = np.array([
                [2, 1, 1],
                [1, 2, 1], 
                [1, 1, 2]
            ]) * area / 12.0
            
            # Local stiffness matrix for P1 elements  
            # K[i,j] = ∫_T ∇φᵢ · ∇φⱼ dx = area * ∇φᵢ · ∇φⱼ
            local_stiffness = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    if i == 0:
                        grad_i = grad1
                    elif i == 1:
                        grad_i = grad2
                    else:
                        grad_i = grad3
                    
                    if j == 0:
                        grad_j = grad1
                    elif j == 1:
                        grad_j = grad2
                    else:
                        grad_j = grad3
                    
                    local_stiffness[i, j] = area * np.dot(grad_i, grad_j)
            
            # Assemble into global matrices
            for i in range(3):
                for j in range(3):
                    M[triangle[i], triangle[j]] += local_mass[i, j]
                    K[triangle[i], triangle[j]] += local_stiffness[i, j]
        
        # Convert to CSR format for efficiency
        self.logger.debug("Converting matrices to CSR format...")
        self.mass_matrix = M.tocsr()
        self.stiffness_matrix = K.tocsr()
        
        self.logger.info(f"Matrix computation completed: "
                        f"Mass matrix {self.mass_matrix.shape}, "
                        f"Stiffness matrix {self.stiffness_matrix.shape}")
        
    def compute_total_area(self):
        """Compute the total area of the ring mesh."""
        if self.mass_matrix is None:
            self.logger.debug("Computing matrices to get total area...")
            self.compute_matrices()
        
        # Sum of mass matrix gives total area
        total_area = np.sum(self.mass_matrix)
        self.logger.debug(f"Total area from mass matrix: {total_area:.6f}")
        return total_area
    
    def get_vertex_count(self):
        """Get the number of vertices in the mesh."""
        return self.vertices.shape[0]
    
    def get_triangle_count(self):
        """Get the number of triangles in the mesh."""
        return self.triangles.shape[0]
    
    def _compute_triangle_statistics(self):
        """Compute basic statistics about the mesh triangles."""
        self.logger.debug("Computing triangle statistics...")
        
        areas = []
        for triangle in self.triangles:
            p1 = self.vertices[triangle[0]]
            p2 = self.vertices[triangle[1]]
            p3 = self.vertices[triangle[2]]
            
            edge1 = p2 - p1
            edge2 = p3 - p1
            area = 0.5 * abs(np.cross(edge1, edge2))
            areas.append(area)
        
        self.triangle_areas = np.array(areas)
        self.min_triangle_area = np.min(areas)
        self.max_triangle_area = np.max(areas)
        self.mean_triangle_area = np.mean(areas)
        
        self.logger.debug(f"Triangle statistics: min={self.min_triangle_area:.6f}, "
                         f"max={self.max_triangle_area:.6f}, mean={self.mean_triangle_area:.6f}")
        
    def get_mesh_statistics(self):
        """Get comprehensive mesh statistics."""
        stats = {
            'n_vertices': self.get_vertex_count(),
            'n_triangles': self.get_triangle_count(),
            'total_area': self.compute_total_area(),
            'theoretical_area': np.pi * (self.r_outer**2 - self.r_inner**2),
            'min_triangle_area': self.min_triangle_area,
            'max_triangle_area': self.max_triangle_area,
            'mean_triangle_area': self.mean_triangle_area,
            'r_inner': self.r_inner,
            'r_outer': self.r_outer,
            'n_radial': self.n_radial,
            'n_angular': self.n_angular
        }
        
        self.logger.info(f"Mesh statistics: {stats['n_vertices']} vertices, "
                        f"{stats['n_triangles']} triangles, "
                        f"area={stats['total_area']:.6f}")
        
        return stats
    
    @property
    def M(self):
        """Get the mass matrix."""
        if self.mass_matrix is None:
            self.logger.debug("Computing matrices to access mass matrix...")
            self.compute_matrices()
        return self.mass_matrix
    
    @property
    def K(self):
        """Get the stiffness matrix."""
        if self.stiffness_matrix is None:
            self.logger.debug("Computing matrices to access stiffness matrix...")
            self.compute_matrices()
        return self.stiffness_matrix
    
    @property
    def v(self):
        """Get the mass matrix column sums (v = 1ᵀM)."""
        if self.mass_matrix is None:
            self.logger.debug("Computing matrices to access mass matrix column sums...")
            self.compute_matrices()
        return np.sum(self.mass_matrix.toarray(), axis=0)
    
    def verify_matrix_properties(self):
        """
        Verify that the computed matrices have the correct properties.
        
        For the Γ-convergence functional J_ε(u) = ε ∫_Ω |∇u|² + (1/ε) ∫_Ω u²(1-u)²:
        - Mass matrix M should satisfy: ∫_Ω u² dx ≈ u^T M u
        - Stiffness matrix K should satisfy: ∫_Ω |∇u|² dx ≈ u^T K u
        """
        if self.mass_matrix is None or self.stiffness_matrix is None:
            self.logger.warning("Matrices not computed yet. Computing matrices...")
            self.compute_matrices()
        
        self.logger.info("=== Matrix Property Verification ===")
        
        # Test 1: Mass matrix should be symmetric and positive definite
        M_dense = self.mass_matrix.toarray()
        K_dense = self.stiffness_matrix.toarray()
        
        self.logger.info(f"Mass matrix properties:")
        mass_symmetric = np.allclose(M_dense, M_dense.T)
        mass_positive_definite = np.all(np.linalg.eigvals(M_dense) > 0)
        
        self.logger.info(f"  Symmetric: {mass_symmetric}")
        self.logger.info(f"  Positive definite: {mass_positive_definite}")
        self.logger.info(f"  Sum of entries: {np.sum(M_dense):.6f} (should equal total area)")
        
        self.logger.info(f"Stiffness matrix properties:")
        stiffness_symmetric = np.allclose(K_dense, K_dense.T)
        stiffness_positive_semidefinite = np.all(np.linalg.eigvals(K_dense) >= 0)
        
        self.logger.info(f"  Symmetric: {stiffness_symmetric}")
        self.logger.info(f"  Positive semi-definite: {stiffness_positive_semidefinite}")
        self.logger.info(f"  Sum of entries: {np.sum(K_dense):.6f} (should be small)")
        
        # Test 2: Verify that constant function has zero gradient energy
        n_vertices = self.vertices.shape[0]
        u_const = np.ones(n_vertices)
        gradient_energy = u_const @ K_dense @ u_const
        self.logger.info(f"Gradient energy of constant function: {gradient_energy:.2e} (should be ~0)")
        
        # Test 3: Verify mass matrix gives correct area
        area_from_mass = np.sum(M_dense)
        theoretical_area = np.pi * (self.r_outer**2 - self.r_inner**2)
        area_error = abs(area_from_mass - theoretical_area) / theoretical_area * 100
        self.logger.info(f"Area from mass matrix: {area_from_mass:.6f}")
        self.logger.info(f"Theoretical area: {theoretical_area:.6f}")
        self.logger.info(f"Area error: {area_error:.4f}%")
        
        return {
            'mass_symmetric': mass_symmetric,
            'mass_positive_definite': mass_positive_definite,
            'stiffness_symmetric': stiffness_symmetric,
            'stiffness_positive_semidefinite': stiffness_positive_semidefinite,
            'constant_gradient_energy': gradient_energy,
            'area_error': area_error
        }
    
    def verify_triangle_orientation(self):
        """
        Verify that all triangles have counterclockwise orientation.
        
        For a triangle with vertices (p1, p2, p3), the orientation is:
        - Counterclockwise if (p2-p1) × (p3-p1) > 0
        - Clockwise if (p2-p1) × (p3-p1) < 0
        """
        self.logger.info("=== Triangle Orientation Verification ===")
        
        orientations = []
        for i, triangle in enumerate(self.triangles):
            p1 = self.vertices[triangle[0]]
            p2 = self.vertices[triangle[1]]
            p3 = self.vertices[triangle[2]]
            
            # Compute cross product to determine orientation
            edge1 = p2 - p1
            edge2 = p3 - p1
            cross_product = np.cross(edge1, edge2)
            
            # Positive cross product means counterclockwise
            is_counterclockwise = cross_product > 0
            orientations.append(is_counterclockwise)
            
            if not is_counterclockwise:
                self.logger.warning(f"Triangle {i} is clockwise!")
        
        counterclockwise_count = sum(orientations)
        total_triangles = len(self.triangles)
        
        self.logger.info(f"Orientation summary:")
        self.logger.info(f"  Total triangles: {total_triangles}")
        self.logger.info(f"  Counterclockwise: {counterclockwise_count}")
        self.logger.info(f"  Clockwise: {total_triangles - counterclockwise_count}")
        self.logger.info(f"  All counterclockwise: {all(orientations)}")
        
        return {
            'all_counterclockwise': all(orientations),
            'counterclockwise_count': counterclockwise_count,
            'total_triangles': total_triangles,
            'orientations': orientations
        } 