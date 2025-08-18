#!/usr/bin/env python3
"""
Minimal‑perimeter equal‑area partitions on a surface via Modica–Mortola relaxation
(Modified to optionally use projected gradient for equality constraints)

Usage examples:
$ python minimal_partitions.py --surface sphere --n 12 --projected_gradient
$ python minimal_partitions.py --surface torus --n 8 --R 1.0 --r 0.6 --projected_gradient
"""
from __future__ import annotations
import argparse
import dataclasses as dc
import math
from typing import Tuple, List
import numpy as np
from numpy.linalg import norm
from scipy.sparse import coo_matrix, csr_matrix
from scipy.linalg import cho_factor, cho_solve  # PG: for small SPD solves
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Mesh utilities
# ------------------------------------------------------------
@dc.dataclass
class Mesh:
    V: np.ndarray  # (N,3) vertices
    T: np.ndarray  # (M,3) faces (indices)
    @property
    def nV(self):
        return self.V.shape[0]
    @property
    def nT(self):
        return self.T.shape[0]

def build_icosphere(subdiv: int = 3, radius: float = 1.0) -> Mesh:
    t = (1.0 + math.sqrt(5.0)) / 2.0
    V = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ], dtype=float)
    F = np.array([
        [0,11,5], [0,5,1], [0,1,7], [0,7,10], [0,10,11],
        [1,5,9], [5,11,4], [11,10,2], [10,7,6], [7,1,8],
        [3,9,4], [3,4,2], [3,2,6], [3,6,8], [3,8,9],
        [4,9,5], [2,4,11], [6,2,10], [8,6,7], [9,8,1]
    ], dtype=int)
    def normalize(X):
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        return X * radius
    V = normalize(V)
    for _ in range(subdiv):
        mid_cache = {}
        def midpoint(i, j):
            key = tuple(sorted((i, j)))
            if key in mid_cache:
                return mid_cache[key]
            m = (V[i] + V[j]) * 0.5
            mid_cache[key] = len(V_list)
            V_list.append(m)
            return mid_cache[key]
        V_list = list(V)
        newF = []
        for a,b,c in F:
            ab = midpoint(a,b)
            bc = midpoint(b,c)
            ca = midpoint(c,a)
            newF += [[a,ab,ca],[b,bc,ab],[c,ca,bc],[ab,bc,ca]]
        V = np.array(V_list, dtype=float)
        V = normalize(V)
        F = np.array(newF, dtype=int)
    return Mesh(V, F)

def build_torus(nu: int = 80, nv: int = 40, R: float = 1.0, r: float = 0.6) -> Mesh:
    us = np.linspace(0, 2*np.pi, nu, endpoint=False)
    vs = np.linspace(0, 2*np.pi, nv, endpoint=False)
    U, Vv = np.meshgrid(us, vs, indexing='ij')
    X = np.empty((nu, nv, 3), dtype=float)
    X[...,0] = (R + r*np.cos(Vv)) * np.cos(U)
    X[...,1] = (R + r*np.cos(Vv)) * np.sin(U)
    X[...,2] = r * np.sin(Vv)
    Vflat = X.reshape(-1,3)
    def idx(i,j):
        return (i % nu)*nv + (j % nv)
    F = []
    for i in range(nu):
        for j in range(nv):
            a = idx(i,j); b = idx(i+1,j); c = idx(i+1,j+1); d = idx(i,j+1)
            F += [[a,b,c],[a,c,d]]
    return Mesh(Vflat, np.array(F, dtype=int))

# ------------------------------------------------------------
# FEM assembly (P1 on triangles)
# ------------------------------------------------------------
def assemble_mass_stiffness(mesh: Mesh) -> Tuple[csr_matrix, csr_matrix, np.ndarray]:
    V, T = mesh.V, mesh.T
    N = mesh.nV
    I=[]; J=[]; Mdata=[]; Sdata=[]
    for (a,b,c) in T:
        xa, xb, xc = V[a], V[b], V[c]
        e1 = xb - xa
        e2 = xc - xa
        nvec = np.cross(e1, e2)
        area = 0.5 * norm(nvec)
        if area == 0:
            continue
        # robust orthonormal basis in triangle plane
        n_hat = nvec / norm(nvec)
        t1 = e1 - n_hat*np.dot(n_hat, e1)
        t1 = t1 / norm(t1)
        t2 = np.cross(n_hat, t1)
        # 2D coords
        xa2 = np.array([0.0, 0.0])
        xb2 = np.array([norm(e1 - n_hat*np.dot(n_hat,e1)), 0.0])
        xc2 = np.array([np.dot(e2, t1), np.dot(e2, t2)])
        x_a, y_a = xa2
        x_b, y_b = xb2
        x_c, y_c = xc2
        denom = 2.0 * area
        grad_a_2d = np.array([y_b - y_c, x_c - x_b]) / denom
        grad_b_2d = np.array([y_c - y_a, x_a - x_c]) / denom
        grad_c_2d = np.array([y_a - y_b, x_b - x_a]) / denom
        grad_a = grad_a_2d[0] * t1 + grad_a_2d[1] * t2
        grad_b = grad_b_2d[0] * t1 + grad_b_2d[1] * t2
        grad_c = grad_c_2d[0] * t1 + grad_c_2d[1] * t2
        local_M = (area/12.0) * np.array([[2,1,1],[1,2,1],[1,1,2]], dtype=float)
        G = np.array([grad_a, grad_b, grad_c])
        local_K = area * (G @ G.T)
        ids = [a,b,c]
        for i in range(3):
            for j in range(3):
                I.append(ids[i]); J.append(ids[j]); Mdata.append(local_M[i,j]); Sdata.append(local_K[i,j])
    M = coo_matrix((Mdata,(I,J)), shape=(N,N)).tocsr()
    K = coo_matrix((Sdata,(I,J)), shape=(N,N)).tocsr()
    lumped_weights = M @ np.ones(N)
    return M, K, lumped_weights

# ------------------------------------------------------------
# Projection onto constraints (Algorithm 1-style)
# ------------------------------------------------------------
def project_partition_and_areas(A: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Orthogonal projection of A (N x nph) so that:
      (i) row sums == 1
     (ii) mass-weighted column sums equal (equal areas), with weights v = M @ 1
    Returns the projected array.
    """
    N, n = A.shape
    e = A.sum(axis=1) - 1.0                 # row-sum residual (N,)
    f = (v[:,None]*A).sum(axis=0) - (v.sum()/n) * np.ones(n)  # column mass residual (n,)
    vv = float(np.dot(v, v))
    C = np.full((n,n), vv/n)
    np.fill_diagonal(C, vv - vv/n)
    q = f - (v @ e) * np.ones(n) / n
    lam = np.zeros(n)
    if n > 1:
        lam[:n-1] = np.linalg.solve(C[:n-1,:n-1], q[:n-1])
    S = lam.sum()
    eta = (e - S * v) / n
    Aorth = eta[:,None] * np.ones((1,n)) + v[:,None] * lam[None,:]
    return A - Aorth

# ------------------------------------------------------------
# PG: Build and apply projector onto tangent of equality constraints
# ------------------------------------------------------------
def build_equality_projector_structs(N: int, nph: int, v: np.ndarray):
    """
    PG: Prepare structures to apply P = I - B^T (B B^T)^{-1} B to a vectorized gradient.
    We avoid explicitly forming B; we implement B and B^T actions using structure.

    Constraints:
      - Row sums per vertex p: sum_i A[p,i] = 1  (N constraints)
      - Equal-area per phase i: sum_p v[p] A[p,i] = const  (nph constraints, drop the last for full rank)

    Let m = N + (nph-1). B maps a = vec(A) in R^{N*nph} to R^m as:
      (Row constraints): (Ba)[p]           = sum_i A[p,i]
      (Area constraints): (Ba)[N + i]      = sum_p v[p] A[p,i], for i=0..nph-2  (drop last)
    """
    m = N + max(0, nph-1)

    # Precompute Cholesky of S = B B^T (size m x m), which is block diagonal plus a rank-1 within the area block.
    # We assemble S explicitly in dense form since m << N*nph typically.
    S = np.zeros((m, m), dtype=float)

    # Block: row constraints vs row constraints: for p,q, <row_p, row_q> = nph if p==q else 0
    # because row_p has ones at entries (p, i) for all i; dot with itself has nph ones.
    for p in range(N):
        S[p, p] = float(nph)

    # Block: area constraints (first nph-1) vs itself:
    # For i,j in 0..nph-2, <area_i, area_j> = (v dot v) * delta_{ij}
    # since area_i selects entries (p,i) weighted by v[p]; area_i and area_j have disjoint supports when i!=j.
    if nph > 1:
        vv = float(v @ v)
        for i in range(nph-1):
            S[N+i, N+i] = vv

    # Block: cross terms row vs area: zero (supports are disjoint in phase index set vs row index sum).
    # So S is diagonal: diag([nph,...,nph]_{N times}, [vv,...,vv]_{(nph-1) times})
    # Hence its Cholesky is trivial.
    diag_S = np.diag(S).copy()
    # Guard against zero vv (degenerate): add tiny regularization
    diag_S[diag_S == 0.0] = 1e-16
    chol = np.sqrt(diag_S)  # since S is diagonal

    # Provide closures for Bx and BTy using shapes
    def B_apply(a_vec: np.ndarray) -> np.ndarray:
        # a_vec is vec(A) with shape (N*nph,)
        A = a_vec.reshape(N, nph, order='C')
        y = np.empty(m, dtype=float)
        # Row sums
        y[:N] = A.sum(axis=1)
        # Area sums for first nph-1 phases
        if nph > 1:
            y[N:] = (v[:,None] * A[:,:nph-1]).sum(axis=0)
        return y

    def BT_apply(y: np.ndarray) -> np.ndarray:
        # y has length m
        # Build an A-like matrix then vectorize
        A_like = np.zeros((N, nph), dtype=float)
        # Row constraints contribute equally to all phases on row p
        # For each p, add y[p] to all phases in row p
        if N > 0:
            A_like += y[:N,None]
        # Area constraints: for i=0..nph-2, add y[N+i]*v to column i
        if nph > 1:
            A_like[:, :nph-1] += v[:,None] * y[N:][None,:]
        return A_like.reshape(N*nph, order='C')

    # Solve S z = rhs via diagonal inverse (or generalized to Cholesky if extended)
    def solve_S(rhs: np.ndarray) -> np.ndarray:
        # Since S is diagonal with entries chol^2, z = rhs / diag(S)
        return rhs / (chol * chol)

    return B_apply, BT_apply, solve_S

def apply_equality_projector_to_grad(G: np.ndarray, B_apply, BT_apply, solve_S) -> np.ndarray:
    """
    PG: Given full gradient G (N x nph), return G_proj = reshape( (I - B^T (BB^T)^{-1} B) vec(G) ).
    """
    N, nph = G.shape
    g = G.reshape(N*nph, order='C')
    y = B_apply(g)              # y = B g
    z = solve_S(y)              # z = (BB^T)^{-1} y
    corr = BT_apply(z)          # B^T z
    g_proj = g - corr
    return g_proj.reshape(N, nph, order='C')

# ------------------------------------------------------------
# Energy, gradient, and solver
# ------------------------------------------------------------
def energy_and_grad(A: np.ndarray, K: csr_matrix, M: csr_matrix, eps: float) -> Tuple[float, np.ndarray]:
    """
    Energy: sum_i [ eps * u_i^T K u_i + (1/eps) * ∫ W(u_i) dA ], with W(u)=u^2(1-u)^2.
    Gradient: 2*eps*K*u_i + (2/eps) * M @ (u_i*(1-u_i)*(1-2u_i)).
    """
    N, n = A.shape
    val = 0.0
    G = np.zeros_like(A)
    ones = np.ones(N)
    for i in range(n):
        u = A[:,i]
        Ku = K @ u
        g = u*u * (1.0 - u)*(1.0 - u)
        val += eps * float(u @ Ku) + (1.0/eps) * float(g @ (M @ ones))
        dgdu = 2.0*u*(1.0-u)*(1.0-2.0*u)
        G[:,i] = 2.0*eps*Ku + (2.0/eps) * (M @ dgdu)
    return val, G

def projected_gradient_descent(mesh: Mesh, nph: int, eps_seq: List[float], steps_per_eps: int = 400,
                               step0: float = 1.0, restarts: int = 3, seed: int = 0,
                               projected_gradient: bool = False) -> Tuple[np.ndarray, List[float]]:
    rng = np.random.default_rng(seed)
    M, K, v = assemble_mass_stiffness(mesh)
    N = mesh.nV

    # PG: Prepare projector structures if requested
    if projected_gradient:
        B_apply, BT_apply, solve_S = build_equality_projector_structs(N, nph, v)

    bestA, bestE = None, float('inf')
    for r in range(restarts):
        # Random simplex initialization then project
        A = rng.random((N, nph))
        A = A / A.sum(axis=1, keepdims=True)
        A = project_partition_and_areas(A, v)

        for eps in eps_seq:
            step = step0
            for it in range(steps_per_eps):
                # Ensure feasibility (equalities) before evaluating
                A = project_partition_and_areas(np.clip(A, 1e-8, 1-1e-8), v)

                E, G = energy_and_grad(A, K, M, eps)

                # PG: project gradient onto tangent of equality constraints if enabled
                if projected_gradient:
                    G_use = apply_equality_projector_to_grad(G, B_apply, BT_apply, solve_S)
                else:
                    G_use = G

                # Backtracking line search (with final equality projection)
                step_try = step
                while True:
                    A_new = A - step_try * G_use
                    # Box handling: clip can break equalities; fix with one projection.
                    A_new = project_partition_and_areas(np.clip(A_new, 1e-8, 1-1e-8), v)
                    E_new, _ = energy_and_grad(A_new, K, M, eps)

                    # Armijo sufficient decrease with ||G_use||_F^2 surrogate
                    if E_new <= E - 1e-4 * step_try * float(np.sum(G_use*G_use)):
                        A, E = A_new, E_new
                        step = step_try  # keep the accepted step as new baseline
                        break
                    step_try *= 0.5
                    if step_try < 1e-6:
                        # Give up on this iteration if step becomes too small
                        break

            # mild sharpening between epsilons (box only; equalities enforced next iter)
            A = np.clip(A, 0.0, 1.0)

        if E < bestE:
            bestA, bestE = A.copy(), E

    return bestA, [bestE]

# ------------------------------------------------------------
# Perimeter estimate and visualization
# ------------------------------------------------------------
def estimate_perimeter(E_eps: float) -> float:
    # Γ‑limit constant: J_ε → (1/3) * Perimeter, hence Per ≈ 3 * J_ε for small ε
    return 3.0 * E_eps

def plot_partition(mesh: Mesh, A: np.ndarray, title: str = ""):
    V, T = mesh.V, mesh.T
    labels = np.argmax(A, axis=1)
    face_labels = labels[T].mean(axis=1)
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(V[:,0], V[:,1], V[:,2], triangles=T, cmap='tab20', linewidth=0.2,
                    antialiased=True, shade=True, alpha=1.0, edgecolor='none', array=face_labels)
    ax.set_box_aspect([1,1,1])
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    plt.show()

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--surface', choices=['sphere','torus'], default='sphere')
    ap.add_argument('--subdiv', type=int, default=3, help='sphere: icosphere subdivisions (≥2 recommended)')
    ap.add_argument('--R', type=float, default=1.0, help='torus major radius')
    ap.add_argument('--r', type=float, default=0.6, help='torus minor radius')
    ap.add_argument('--nu', type=int, default=96, help='torus u-samples')
    ap.add_argument('--nv', type=int, default=48, help='torus v-samples')
    ap.add_argument('--n', type=int, default=12, help='number of equal-area cells')
    ap.add_argument('--eps', type=str, default='0.08,0.06,0.04,0.03,0.02', help='comma-sep epsilon schedule (largest→smallest)')
    ap.add_argument('--steps', type=int, default=350, help='gradient steps per epsilon')
    ap.add_argument('--restarts', type=int, default=2, help='random restarts')
    ap.add_argument('--seed', type=int, default=0)
    # PG: new flag to enable projected gradient on equalities
    ap.add_argument('--projected_gradient', action='store_true', help='use projected gradient on equality constraints')

    args = ap.parse_args()
    if args.surface == 'sphere':
        mesh = build_icosphere(args.subdiv, 1.0)
        surf_name = f"Sphere (subdiv={args.subdiv})"
    else:
        mesh = build_torus(args.nu, args.nv, args.R, args.r)
        surf_name = f"Torus (R={args.R}, r={args.r})"

    eps_seq = [float(x) for x in args.eps.split(',')]
    A, energies = projected_gradient_descent(
        mesh, args.n, eps_seq, steps_per_eps=args.steps,
        step0=1.0, restarts=args.restarts, seed=args.seed,
        projected_gradient=args.projected_gradient
    )
    E = energies[-1]
    per = estimate_perimeter(E)
    print(f"Surface: {surf_name}")
    print(f"n = {args.n},  final epsilon = {eps_seq[-1]:.4f}")
    print(f"Relaxed energy J_eps = {E:.6f}")
    print(f"Estimated total perimeter ≈ 3*J_eps = {per:.6f}")
    if not hasattr(args, 'no_plot') or not args.no_plot:
        plot_partition(mesh, A, title=f"{surf_name} — n={args.n}")

if __name__ == '__main__':
    main()
