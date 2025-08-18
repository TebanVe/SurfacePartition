# Partitions of Minimal Length on Manifolds

**Beniamin Bogosel and Édouard Oudet**

## Abstract

We study partitions on three dimensional manifolds which minimize the total geodesic perimeter. We propose a relaxed framework based on a Γ-convergence result and we show some numerical results. We compare our results to those already present in the literature in the case of the sphere. For general surfaces we provide an optimization algorithm on meshes which can give a good approximation of the optimal cost, starting from the results obtained using the relaxed formulation.

## 1. Introduction

In this article we propose a theoretical and numerical framework for the study of the partitions $(ω_i)_{i=1}^n$ of a surface $S ⊂ \mathbb{R}^3$ which minimize the total geodesic perimeter while keeping a prescribed area for each cell. Thus, we are interested in minimizing $\mathcal{H}^1(\cup_{i=1}^n ∂_S ω_i)$ or equivalently

$$\text{Per}(ω_1) + ... + \text{Per}(ω_n)$$

in the class of partitions $(ω_i)$ of the surface $S$ such that $|ω_i| = c_i$, with the compatibility constraint $c_1 + ... + c_n = |S|$. Here $∂_S ω$ denotes the boundary of a set $ω$ as a subset of the surface $S$, $\text{Per}(ω)$ denotes the geodesic perimeter of $ω$, i.e. the perimeter of $ω$ regarded as a subset of the surface $S$ and $|ω|$ is the area of the subset $ω$. General theoretical results concerning these minimal partitioning problems are presented by Morgan in [16]. This theoretical result states that boundaries of a minimal-perimeter partition are arcs of constant geodesic curvature and the boundaries of the sets meet in threes with angles of measure $2π/3$.

The more specific case concerning the minimal perimeter partitions of sphere with cells of equal areas was intensively studied from both theoretical and numerical points of view. In the case $n = 2$ the solution is the partition into two half-spheres. This was proved by Bernstein in 1905 [5]. In the case $n = 3$ the optimal candidate is the partition of the sphere into three slices corresponding to an angle of $2π/3$. This was proved by Masters in [15]. The case $n = 12$ was solved by Hales in [13] using methods similar to the ones involved in the proof of the honeycomb conjecture [12]. The case $n = 4$ was treated by Engelstein in [11] and the corresponding optimal partition is the one associated to the regular tetrahedron.

The case of the sphere has been studied numerically by Cox and Flikkema [9] using the Surface Evolver software [7]. They perform computations for $n ∈ ⟦2, 32⟧$ and they confirm the natural conjecture for $n = 6$: the optimal partition in this case is probably the one associated to the cube. Their algorithm performs the perimeter optimization after choosing a topological structure for the partition. Thus, the optimization algorithm has to know a priori the topological structure in order to find the corresponding local minimum. In the end we keep the configuration which gives the best optimal cost among the admissible combinatorial possibilities.

The algorithm we propose is a generalization of the ideas in [17] to the case of surfaces. First, there is a theoretical result, similar to the theorem of Modica and Mortola, which we present in Section 2. This theoretical result justifies the use of the functional

$$J_ε(u) = ε \int_S |∇_τ u|^2 + \frac{1}{ε} \int_S u^2(1 - u)^2$$

as an approximation of the perimeter as $ε → 0$. The direct consequence of the Γ-convergence result is that a sequence of minimizers $u_ε$ for $J_ε$ under the constraint $\int_S u_ε = c$ converges to a minimizer of the geodesic perimeter under area constraint. For the partitioning case we prove that functionals of the type

$$\sum_{i=1}^n J_ε(u_i)$$

approximate the perimeter as $ε → 0$, where $u_i$ are functions associated to the sets $ω_i$ which satisfy some integral and non-overlapping constraints. We implement an optimization algorithm which is able to solve the above problem on a large class of surfaces. This is an advantage over the methods used in [9] which can be used only in the case of the sphere.

Working with the relaxed formulation does not provide an exact representation of the contours. Thus, we cannot directly provide the associated cost once we have the relaxed optimal partitions. The particular case of the sphere can be solved directly by noting that boundaries between two cells have constant geodesic curvature [16] and are, thus, arcs of circles. We recover all the results presented in [9] in the case of the sphere. On more complex surfaces it is complicated to explicitly work with curves of constant geodesic curvature. Nevertheless, we can extract the contours from the density representation in order to compute the total perimeter. Since the extracted contours are not smooth, we perform a constrained optimization stage on the triangulated surface preserving the topology to obtain reliable approximations of the optimal costs.

## 2. Theoretical Result

As in [17] we would like to have a rigorous theoretical framework which justifies our numerical method. In the euclidean case it was an adapted version of the Modica-Mortola theorem to the case of partitions which provided the needed result. In the case of surfaces we did not find an equivalent result in the literature. We did find the results in [4] which suggest that the relaxation we consider is the right one on general manifolds. In the above reference a the authors do not prove a Γ-convergence result, but only the convergence of minimisers. We are concerned here only with smooth manifolds of codimension one and in this particular case it is possible to adapt classical methods in order to prove a Γ-convergence result.

We start by defining the space of functions of bounded variations on a $d - 1$ dimensional surface in $\mathbb{R}^d$. Let $S$ be a smooth $d - 1$ dimensional manifold without boundary in $\mathbb{R}^d$. In the following we consider the tangential gradient of a function $u$ defined on $S$ to be

$$∇_τ u = ∇\tilde{u} - (∇\tilde{u} \cdot n)n,$$

where $\tilde{u}$ is a regular extension of $u$ in a neighbourhood of $S$ and $n$ denotes the normal vector to the surface. In the same way we define the tangential divergence of a vector field $w ∈ C^1(S; \mathbb{R}^d)$ by

$$\text{div}_τ w = \text{tr}(D_τ w)$$

where the matrix $D_τ w$ contains on line $i$ the tangential gradient of the $i$-th component of $w$, i.e. $∇_τ w_i$. See [14, Section 5.4] for further details.

We consider the space of functions with bounded variation on $S$

$$BV(S) = \{u ∈ L^1(S) : TV(u) < ∞\}$$

where

$$TV(u) = \sup\{\int_S u \text{div}_τ g : |g|_∞ ≤ 1\}.$$

Using the divergence theorem on manifolds (see [14, Section 5.4]), we obtain that if $u$ is $C^1(S)$ then

$$TV(u) = \int_S |∇_τ u|.$$

If $ω$ is a subset of $S$ we define its generalized perimeter as $\text{Per}(ω) = TV(χ_ω)$, where $χ_ω$ represents the characteristic function of $ω$. By mimicking the proof in the euclidean case we can prove that the total variation is lower semi-continuous for the $L^1(S)$ convergence. We refer to [6] for more details.

Let $(C_i)$ is a set of local charts which cover $S$ such that each $C_i$ is diffeomorphic to a connected and bounded open subset $D_i$ of $\mathbb{R}^{d-1}$. We denote by $θ_i: D_i → C_i$ these diffeomorphisms. Then it is possible to transfer a function $u$ from $C_i$ to $D_i$ using the transformation $\tilde{u}_i = u ∘ θ_i$. These new functions $\tilde{u}_i$, which lie now in Euclidean spaces, are functions of bounded variation. Therefore, it is possible to transfer some of the theory of BV functions from Euclidean spaces to manifolds of codimension 1 by using local charts and partitions of unity. In particular, it is possible to approximate finite perimeter sets $ω ⊂ S$ with smooth sets $ω_n ⊂ S$ such that $ω_n → ω$ in the $L^1(S)$ topology and $\text{Per}(ω_n) → \text{Per}(ω)$.

We are now ready to state the relaxation result in the case of a single phase, which will be generalized later to the case of a partition. To derive the theorem below we follow the approach provided by Buttazzo in [8] and Alberti in [1].

**Theorem 2.1.** Define $F_ε, F : L^1(S) → [0, +∞]$ as follows:

$$F_ε(u) = \begin{cases}
\int_S \left( ε|∇_τ u|^2 + \frac{1}{ε}u^2(1 - u)^2 \right) dσ & \text{if } u ∈ H^1(S), \int_S u = c \\
+∞ & \text{otherwise}.
\end{cases}$$

$$F(u) = \begin{cases}
\frac{1}{3}\text{Per}(\{u = 1\}) & \text{if } u ∈ BV(S, \{0, 1\}), \int_S u = c \\
+∞ & \text{otherwise}.
\end{cases}$$

Then $F_ε \xrightarrow{Γ} F$ in the $L^1(S)$ topology.

*Proof:* We define $φ(t) = \int_0^t |s(1 - s)|ds$. We consider a sequence $(u_ε) → u$ in $L^1(S)$ such that $\liminf_{ε→0} F_ε(u_ε) < +∞$. Since $F_ε(u_ε) ≥ \frac{1}{ε}\int_S u_ε^2(1 - u_ε)^2$, if we take a subsequence of $u_ε$ which converges almost everywhere to $u$ we obtain that

$$\int_S u^2(1 - u)^2 = 0,$$

and thus $u ∈ \{0, 1\}$ almost everywhere in $S$. Note that truncating $u_ε$ between 0 and 1 decreases the value of $F_ε(u_ε)$ while preserving the fact that $u_ε → u$ in $L^1(S)$. Also note that $φ$ is Lipschitz on $[0, 1]$ so we can conclude that $φ ∘ u_ε → φ ∘ u$ in $L^1(S)$. By applying the classical inequality $a^2 + b^2 ≥ 2ab$ we get that

$$F_ε(u_ε) ≥ 2 \int_S |∇_τ u|φ'(u_ε) = 2 \int_S |∇_τ (φ ∘ u_ε)|.$$

Taking $\liminf$ in the above inequality and using the semi-continuity of the total variation with respect to the $L^1(S)$ convergence we obtain that

$$\liminf_{ε→0} F_ε(u_ε) ≥ 2TV(φ ∘ u) = 2φ(1)TV(u).$$

Since $u$ is a characteristic function, it follows that the perimeter of $\{u = 1\}$ is bounded and therefore $u ∈ BV(S, \{0, 1\})$. Note that $φ(1) = 1/6$ and thus we recover the desired constant in front of the perimeter. It is obvious that the integral condition is also preserved in the limit. This concludes the proof of the $Γ - \liminf$ part of the theorem.

For the $Γ - \limsup$ part we need to exhibit a recovery sequence for each $u$ such that $F(u) < +∞$. By a classical argument it is enough to find a recovery sequence only for functions $u$ which are characteristic functions of smooth sets in $S$. See [6] for more details concerning the reduction to regular sets and [3, Theorem 3.42] for the BV approximation of finite perimeter sets with smooth sets.

Let's consider now $u = χ_ω$ where $ω ⊂ S$ is a set with smooth boundary relative to $S$. We consider the signed distance function $d_ω : S → \mathbb{R}$ defined by

$$d_ω(x) = d_τ(x, S \setminus ω) - d_τ(x, ω),$$

where $d_τ$ is the geodesic distance on $S$. Note that $d_ω$ is positive outside $ω$ and negative inside. Consider the optimal profile problem

$$c = \min\{\int_{\mathbb{R}} (W(v) + |v'|^2 : v(-∞) = 0, v(+∞) = 1\}.$$

Any solution of this minimizing problem satisfies $v' = \sqrt{W(v)}$ and we can impose the initial condition $v(0) = 1/2$ in order to have a symmetric behaviour. We can see that the optimal value is $c = 2\int_0^1 \sqrt{W(s)}ds$. In our problem we have chosen $W(s) = s^2(1 - s)^2$. In order to have a function which goes from 0 to 1 in finite time we may choose

$$v^η = \min\{\max\{0,(1 + 2η)v - η\}, 1\}.$$

We see that as $η → 0$ we have

$$c^η = \int_{\mathbb{R}} (W(v^η) + |(v^η)'|^2) → c \text{ as } η → 0.$$

All these considerations are inspired from [6]. We can define

$$u_ε(x) = v^η(d_ω(x)/ε).$$

We can see that

$$F_ε(u_ε) = \int_S \left( ε|∇_τ u|^2 + \frac{1}{ε}W(u) \right)$$

$$= \int_{-T/ε}^{T/ε} \int_{d_ω(x)=t} \left( ε|(v^η)'(d_ω(x)/ε)|^2 \frac{|∇_τ d_ω(x)|^2}{ε^2} + \frac{1}{ε}W(v^η(d_ω/ε)) \right) d\mathcal{H}^{d-2}(x)dt$$

$$= \int_{-T/ε}^{T/ε} \int_{d_ω(x)=t} \frac{1}{ε}(|(v^η)'(t/ε)|^2 + W(v^η(t/ε)))d\mathcal{H}^{d-2}(x)dt$$

$$= \int_{-T/ε}^{T/ε} \text{Per}(d_ω(x) = t)\frac{1}{ε}(|(v^η)'(t/ε)|^2 + W(v^η(t/ε)))dt$$

$$= \int_{-T}^{T} \text{Per}(d_ω(x) = tε)(|(v^η)'(t)|^2 + W(v^η(t)))dt$$

where we have applied the co-area formula and $T$ is chosen such that the support of $v^η$ is inside $[-T, T]$. Since $\lim_{s→0} \text{Per}(\{d_ω(x) = s\}) = \text{Per}(ω)$ we see that for $ε$ small enough there exists $δ$ such that $\text{Per}(d_ω(x) = s) < \text{Per}(ω) + δ$ when $|s| < Tε$. Therefore

$$\limsup_{ε→0} F_ε(u_ε) ≤ (\text{Per}(ω) + δ) \int_{-T}^{T} (|(v^η)'(t)|^2 + W(v^η(t)))dt = (\text{Per}(ω) + δ)c^η.$$

Since this is true for any $δ, η$ small enough, by letting $δ, η → 0$ we obtain the desired result.

In order to have a fixed integral equal to $\int_S χ_ω = c$ it is enough to consider a shift in the definition of $u_ε$:

$$u_ε(x) = v^η((d_ω(x) + s_ε)/ε),$$

where $s_ε ∈ [-Tε, Tε]$. We can see that for $s_ε = Tε$ we have $u_ε = 1$ on $ω$ and thus $\int_S u_ε > c$ while for $s_ε = -Tε$ the support of $u_ε$ is included in $ω$ and we have the opposite inequality. Thus, for each $ε$ small enough we can change the definition of $u_ε$ so that $\int_S u_ε = c$. The estimates presented above are carried with no difficulty in this setting. □

We can now state the result in the partitioning case. We denote by $\mathbf{u}$ an element in $(L^1(S))^n$. In order to simplify the notations we introduce the space

$$X = \left\{\mathbf{u} ∈ (L^1(S))^n: \int_S u_i = c_i, \sum_{i=1}^n u_i = 1\right\}$$

where $c_i$ satisfy the compatibility condition $\sum_{i=1}^n c_i = \mathcal{H}^{d-1}(S)$. It is easy to see that $X$ is closed under the convergence in $(L^1(S))^n$.

**Theorem 2.2.** Define $F_ε, F : (L^1(S))^n → [0, +∞]$ as follows:

$$F_ε(\mathbf{u}) = \begin{cases}
\sum_{i=1}^n \int_S \left( ε|∇_τ u_i|^2 + \frac{1}{ε}u_i^2(1 - u_i)^2 \right) dσ & \text{if } \mathbf{u} ∈ (H^1(S))^n ∩ X \\
+∞ & \text{otherwise}
\end{cases}$$

$$F(\mathbf{u}) = \begin{cases}
\frac{1}{3}\sum_{i=1}^n \text{Per}(\{u_i = 1\}) & \text{if } \mathbf{u} ∈ (BV(S, \{0, 1\}))^n ∩ X \\
+∞ & \text{otherwise}
\end{cases}$$

Then $F_ε \xrightarrow{Γ} F$ in the $(L^1(S))^n$ topology.

*Proof:* It is easy to see that the $Γ - \liminf$ part follows at once from Theorem 2.1 and from the fact that $X$ is closed under the topology of $(L^1(S))^n$.

In order to construct the recovery sequence we reduce the problem to the case where the limit $\mathbf{u}$ consists of piecewise smooth parts in $S$. In this case we define $u_i = v^η(d_{ω_i}(x)/ε)$ as in the one phase case. Thus on each $ω_i$ we have $u_i ≥ 1/2$ which implies that $\sum_{i=1}^n u_i ≥ 1/2$. There are two points which need to be addressed:

(1) The sum equal to 1 condition. Due to the symmetry of the optimal profile we deduce that there is only one zone where the sum condition is not satisfied and that is in the neighborhood of singular points. Since an $ε$-neighborhood of the singular set is of order $ε^{d-1}$. Replacing each $u_i$ by $u_i/(\sum_{i=1}^n u_i)$ in these problematic regions we preserve the regularity of each $u_i$ and we note that the functions have bounded gradient of order $O(1/ε)$. We immediately find that the corresponding energy

$$\int_{N_ε} \left( ε|∇_τ u_i|^2 + \frac{1}{ε}u_i^2(1 - u_i)^2 \right)$$

vanishes as $ε → 0$.

(2) We also need to modify the functions $u_i$ so that they have the same integral over $S$. In order to do this we apply a procedure found in [2] where we consider a family of balls in regions where $u_i ∈ \{0, 1\}$. On each such ball we can consider modifications of $u_i$ such that the sum is preserved and the integrals have the right value. As above, the sum of energies on these balls will be negligible in the limit.

Once these points are addressed, the $\limsup$ estimates follows just like in the one dimensional case and the proof of the theorem is completed. □

## 3. Finite Element Framework

We wish to use this relaxation by Γ-convergence to perform numerical computations so we need a framework which allows us to compute the quantity

$$ε \int_S |∇_τ u|^2 + \frac{1}{ε} \int_S u^2(1 - u)^2,$$

in fast, efficient way. In order to do this we triangulate the surface $S$ and we compute the mass matrix $M$ and the stiffness matrix $K$ associated to the P1 finite elements on this triangulation. Then, if for the sake of simplicity, we use the same notation $u$ for the P1 finite element approximation of $u$, we have

$$\int_S |∇_τ u|^2 = u^T K u$$

and

$$\int_S u^2(1 - u)^2 = v^T M v,$$

where $v = u.^2 \times (1 - u).^2$. We have used the Matlab convention that adding a point before an operation means that we are doing component-wise vector computations. Note that once the matrices $K, M$ are computed, we only have to perform matrix-vector multiplications, which is really fast. In this setting we use the discrete gradients of the above expressions given by:

$$∇_u u^T K u = 2K u,$$
$$∇_u v^T M v = 2M v \times (1 - 2u).$$

The partition condition and the equal areas constraint are imposed by making an orthogonal projection on the linear constraints as follows. We write the discrete vectors representing P1 discretization of the density functions in the following matrix form

$$M = (φ^1 \quad φ^2 \quad ... \quad φ^n).$$

The partition constraint implies that the sum of the elements on every line of $M$ is equal to 1 and the equal area constraint implies that for every column of the matrix $M$ we have the relation $⟨v, φ^i⟩ = A/n$, where $v = 1_{1×N} · M$.

Here the constant $A$ is the total area of the surface, $N$ is the total number of points in the triangulation and the notation $1_{p×q}$ represents the $p × q$ matrix whose entries are all equal to 1. These conditions are discretizations in the finite element setting of the conditions that the integrals of the density functions $u_i$ are all equal to $A/n$. Indeed, given a triangulation $\mathcal{T}$ of $S$ and its associated mass matrix $M$, we have $\int_S 1 · u_i = 1_{1×N} · M · φ^i$, where $φ^i$ is the vector containing the values of $u_i$ at the vertices of the triangulation. The projection routine can be found in Algorithm 1.

**Algorithm 1** Orthogonal projection on the partition and area constraints

**Require:** $A = (a_{ij}) ∈ \mathbb{R}^{N×n}, c ∈ \mathbb{R}^{1×n}, d ∈ \mathbb{R}^{N×1}, v$
1. $(e_i) = \sum_j a_{ij} - c_i$ (line sum error; $N × 1$ column vector)
2. $(f_i) = \sum_i v_i a_{ij} - d_j$ (column scalar product error; $n × 1$ column vector)
3. Define the matrix $C$ of size $n × n$ by
   $$c_{kl} = \begin{cases}
   \|v\|_2^2/n & \text{if } k ≠ l \\
   \|v\|_2^2 - \|v\|_2^2/n & \text{if } k = l
   \end{cases}$$
4. $(q_j) = (f_j) - ⟨v, e⟩/n$ ($n × 1$ column vector)
5. Compute $(λ_j) ∈ \mathbb{R}^{n×1}$ with $λ_n = 0$ such that $C|_{(n-1)×(n-1)}(λ_j)|_{n-1} = (q_j)|_{n-1}$. The indices indicate a sub-matrix with the first $n - 1$ lines and columns, or the sub-vector formed by the first $n - 1$ components.
6. $S = \sum_j λ_j$
7. $η_i = (e_i - S · v_i)/n$ ($N × 1$ column vector)
8. $A_{orth} = (η_i) · 1_{1×n} + v · (λ_j)^T$, where $1_{p×q}$ is the $p × q$ matrix with all entries equal to 1
9. $A = A - A_{orth}$

**return** $A$

Once we have this discrete formulation we use an optimized LBFGS gradient descent procedure [19] to compute the numerical minimizers. In order to avoid local minima where one of the phases $φ^l$ is constant, which arise often when the number of phases is greater than 5, we add a Lagrange multiplier which penalizes the constant functions. In this way, we optimize

$$\sum_{i=1}^n ε \int_S |∇_τ φ^i|^2 + \frac{1}{ε} \int_S (φ^i)^2(1 - φ^i)^2 + λ(\text{std}(φ^i) - s_{target})^2,$$

where $\text{std}(φ^l)$ is the standard deviation of $φ^l$ and $s_{target}$ is the standard deviation of a characteristic function of area $\text{Area}(S)/n$.

In order to have a good approximation of the optimal partition, we want do decrease $ε$ so that the width of the interface is small. We notice that if we chose $ε$ of the same order as the sides of the mesh triangles the algorithm converges. Furthermore, we cannot make $ε$ smaller, since then the gradient term will not contain any real information, as the width of the interface is of size $ε$. In order to avoid this problem, we consider refined meshes associated to each $ε$. At each step where we decrease $ε$ we interpolate the values of the previous optimizer on a refined mesh and we consider these interpolated densities as starting point for the descent algorithm on the new mesh. In the case of the sphere we make four refinements ranging from 10000 to 160000 points. Some optimal configurations, in the case of the sphere, are presented in Figure 1. A detailed study of the case of the sphere along with a comparison with the known results of Cox and Flikkema [9] are presented in the next section.

As underlined before, our approach allows a direct treatment of any surface, as long as a qualitative triangulation is found. We perform some numerical computations on various shapes like a torus, a double torus, and a more complex surface called Banchoff-Chmutov of order 4. A few details about the definitions of these surfaces are provided below:

• We consider a torus of outer radius $R = 1$ and inner radius 0.6 (see Figure 2). This torus is defined as the zero level set of the function
  $$f(x, y, z) = (x^2 + y^2 + z^2 + R^2 - r^2)^2 - 4R^2(x^2 + y^2).$$

• The double torus used in the computation (see Figure 3) is given by the zero level set of the function
  $$f(x, y, z) = (x(x - 1)^2(x - 2) + y^2)^2 + z^2 - 0.03.$$

• The complex Banchoff-Chmutov surface (see Figure 4) is given by the zero level set of the function
  $$f(x, y, z) = T_4(x) + T_4(y) + T_4(z),$$
  where $T_4(X) = 8X^4 - 8X^2 + 1$ is the Tchebychev polynomial of order 4.

## 4. Refined Optimization in the Case of the Sphere

The costs associated to the relaxed functional do not provide a good enough approximation of the total length of the boundaries. In this section we propose a method to approximate the optimal cost in the case of the sphere. The results of [16] state that boundaries of the cells of the optimal partitions have constant geodesic curvature. In the case of the sphere the only such curves are the arcs of circle. See for example [18, Exercise 2.4.9] for a proof. The results of Cox and Flikkema [9] show that optimal configurations are not made of geodesic polygons. In order to perform an optimization procedure which captures this effect they chose to make an initial optimization in the class of geodesic polygons and then divide each geodesic arc into 16 smaller arcs and restart the procedure with more variable points. They manage to approximate well enough the general optimal structure but they still work in the class of geodesic polygons with additional vertices. Our approach presented below is different in the sense that we consider general circle arcs (not necessarily geodesics) which connect the points.

The first step is to extract the topology of the partition from the previous density results, i.e. locate the triple points, the edge connections and construct the faces. In order to perform the refined optimization procedure we need to be able to compute the areas of portions of the sphere determined by arcs of circles. This is possible using the Gauss-Bonnet formula. If $M$ is a smooth subset of a surface then

$$\int_M K dA + \int_{∂M} k_g = 2πχ(M), \quad (4.1)$$

where $K$ is the curvature of the surface, $k_g$ is the geodesic curvature and $χ(M)$ is the Euler characteristic of $M$. This result extends to piecewise smooth curves and in this case we have

$$\int_M K dA + \int_{∂M} k_g + \sum θ_i = 2πχ(M), \quad (4.2)$$

where $θ_i$ are the turning angles between two consecutive smooth parts of the boundary. In the case of a polygon the turning angles are the external angles of the polygon. The formula (4.2) allows the computation of the area of a piece of the sphere bounded by arcs of circle. In this case the Euler characteristic is equal to 1, the curvature of the unit sphere is $K = 1$ and the geodesic curvature is piecewise constant. For more details we refer to [10, Chapter 4].

A first consequence of the Gauss-Bonnet theorem in connection to our problem is noting the fact that, apart from cases where we have a certain symmetry like $n ∈ \{3, 4, 6, 12\}$ the optimal cells are not geodesic polygons. This is made clear in cases where we have a hexagonal cell. If the arcs forming the boundary of such a hexagonal cell would be geodesic polygons then its area would be equal to $6 · 2π/3 - 4π = 0$. Thus a spherical shape bounded by six arcs of circle can never be a geodesic polygon without being degenerate.

In order to perform the optimization we take the vertices as variables and we add one supplementary vertex for each edge. This is enough to contain all the necessary information since an arc of circle is well defined by three distinct points on the sphere. In the sequel we denote $\mathcal{P}_n$ the set of partitions of the sphere into $n$ cells and with $\mathcal{A}_n$ the partitions in $\mathcal{P}_n$ having equal areas. In order to have a simpler numerical treatment of the problem we can incorporate the area constraints in the functional by defining for every partition $(ω_i) ∈ \mathcal{P}_n$ the quantity defined for every $ε > 0$ by

$$G_ε((ω_i)) = \sum_{i=1}^n \text{Per}(ω_i) + \frac{1}{ε} \sum_{i=1}^{n-1} \sum_{j=i+1}^n (\text{Area}(ω_i) - \text{Area}(ω_j))^2.$$

If we denote

$$G((ω_i)) = \begin{cases}
\sum_{i=1}^n \text{Per}(ω_i) & \text{if } (ω_i) ∈ \mathcal{A}_n \\
∞ & \text{if } (ω_i) ∈ \mathcal{P}_n \setminus \mathcal{A}_n.
\end{cases}$$

then we have the following Γ-convergence result.

**Theorem 4.1.** We have that $G_ε \xrightarrow{Γ} G$ for the $L^1(S^2)$ convergence of sets.

*Proof:* For the (LI) property consider a sequence $(ω_i^ε) ⊂ \mathcal{P}_n$ which convergence in $L^1(S^2)$ to $(ω_i)$. It is clear that we have $\text{Area}(ω_i^ε) → \text{Area}(ω_i)$ and the perimeter is lower semicontinuous for the $L^1$ convergence. Thus we have two situations. If $(ω_i) ∈ \mathcal{P}_n \setminus \mathcal{A}_n$ then $\lim_{ε→0} G_ε((u_i^ε)) = ∞$. If $(ω_i) ∈ \mathcal{A}_n$ then the lower semicontinuity of the perimeter implies that $\liminf_{ε→0} G_ε((ω_i^ε)) ≥ G((ω_i))$.

The (LS) property is immediate in this case. Choose $(ω_i) ∈ \mathcal{A}_n$, or else there is nothing to prove. We may choose the recovery sequence equal to $(ω_i)$ for every $ε > 0$. Thus the property is verified immediately. □

**Remark 4.2.** We note that in the above proof the simplicity of the proof of the (LS) property is due to the fact that the functionals $G_ε$ are well defined on the space $\{G < ∞\}$, which makes possible the choice of constant recovery sequences. This is not the case in the results proved in Section 2.

This Γ-convergence result proves that minimizers of $G_ε$ converge to minimizers of $G$. As a consequence, in the numerical computations, we minimize $G_ε$ for $ε$ smaller and smaller in order to approach the minimizers of $G$, which are in fact the desired solutions to our problem.

Since the parameters are of two types: triple points and edge points, we prefer to use an optimization algorithm which is not based on the gradient. The algorithm is described below.

• For each point $P$ consider a family of $m$ tangential directions $(v_i)_{i=1}^m$ chosen as follows: the first direction is chosen randomly and the rest are chosen so that the angles between consecutive directions are $2π/m$.
• Evaluate the cost function for the new partition obtained by perturbing the point $P$ in each of the directions $v_i$ according to a parameter $ε$.
• Choose the direction which has the largest decrease and update the partition accordingly.
• Do the same procedure for each edge point by performing the two possible orthogonal perturbations of the point with respect to the edge.
• If there is no decrease for each of the points of the partition, then decrease $ε$.

This algorithm converges in each of the test cases and the results are presented in Table 1. In the optimization procedure we start with $ε = 1$ and we reiterate the optimization decreasing $ε$ by a factor of 10 at each step until we reach the desired precision on the area constraints. We are able to recover the same results as Cox and Flikkema for $n ∈ [4, 32]$. Furthermore, unlike in the case of geodesic polygons, all triple points consist of boundaries which meet at equal angles of measure $2π/3$. In Figure 5 you can see the results for $n = 9$ and $n = 20$. The red arcs are geodesic connecting the points and are drawn to visually see that not all the boundaries of the optimal structure are geodesic arcs.

**Table 1.** Comparison between our results and the results of Cox and Flikkema in the case of the sphere.

| N | our results |  | Cox-Flikkema |
|---|-------------|-----------|--------------|
|   | non-geo.    | area tol. | non-geo.     |
| 4 | 11.4637     | 5 × 10⁻⁷  | 11.464       |
| 5 | 13.4304     | 2 × 10⁻⁷  | 13.430       |
| 6 | 14.7715     | 2 × 10⁻⁷  | 14.772       |
| 7 | 16.3519     | 3 × 10⁻⁷  | 16.352       |
| 8 | 17.6927     | 3 × 10⁻⁷  | 17.692       |
| 9 | 18.8504     | 2 × 10⁻⁷  | 18.850       |
| 10| 19.9997     | 4 × 10⁻⁷  | 20.000       |
| 11| 21.1398     | 4 × 10⁻⁷  | 21.140       |
| 12| 21.8918     | 5 × 10⁻⁷  | 21.892       |
| 13| 23.0953     | 4 × 10⁻⁷  | 23.095       |
| 14| 23.9581     | 3 × 10⁻⁷  | 23.958       |
| 15| 24.8821     | 2 × 10⁻⁷  | 24.882       |
| 16| 25.7269     | 2 × 10⁻⁷  | 25.727       |
| 17| 26.6365     | 3 × 10⁻⁷  | 26.637       |
| 18| 27.4647     | 2 × 10⁻⁷  | 27.465       |
| 19| 28.2735     | 2 × 10⁻⁷  | 28.274       |
| 20| 28.9992     | 1 × 10⁻⁷  | 28.999       |
| 21| 29.7748     | 2 × 10⁻⁷  | 29.775       |
| 22| 30.5094     | 2 × 10⁻⁷  | 30.509       |
| 23| 31.2260     | 2 × 10⁻⁷  | 31.226       |
| 24| 31.9117     | 3 × 10⁻⁷  | 31.912       |
| 25| 32.6172     | 8 × 10⁻⁸  | 32.617       |
| 26| 33.2675     | 2 × 10⁻⁷  | 33.268       |
| 27| 33.8968     | 9 × 10⁻⁸  | 33.897       |
| 28| 34.5521     | 4 × 10⁻⁷  | 34.552       |
| 29| 35.2065     | 6 × 10⁻⁷  | 35.207       |
| 30| 35.8199     | 5 × 10⁻⁷  | 35.820       |
| 31| 36.3941     | 4 × 10⁻⁶  | 36.394       |
| 32| 36.9310     | 4 × 10⁻⁶  | 36.931       |

Thus we can conclude that the relaxed formulation presented in the previous section is able to match the best known configurations in the literature. Furthermore for $n ∈ [5, 25] ∪ \{32\}$ the algorithm finds the good configuration without much effort, while for $n ∈ [26, 31]$ multiple tries with different initial conditions were needed in order to find the best configuration. The fact that the structure of the partition is not fixed is a great advantage offered by our method.

## 5. Computing the Optimal Cost - General Surfaces

The approach used in the previous section cannot be applied to other surfaces than the sphere. Indeed, the general expression of curves of constant curvature is not known explicitly for other types of surfaces. One way to approximate the total perimeter of the partition would be to extract the contours of the optimal densities and evaluate the length of each discrete contour. A natural way to extract a contour corresponding to a density function would be taking a level set, for example the level 0.5. It is possible to extract such level sets by looking at which triangles contain values which are both above and below the level set. On each triangle which is cut by the contour we make a linear interpolation which determines a segment in the contour of the level set.

Once we have an idea on how to extract the contours, the first question arises: how to make sure that the level sets extracted form a partition of $S$? We denote by $\mathcal{T}$ a triangulation of $S$. If we think of extracting the 0.5 levels of each density, the shapes determined by these contours will not overlap, but around triple points there will be some free space left. One way to make sure that we have extracted a partition is to take the 0.5 levels of the function defined on the triangulation $\mathcal{T}$ by

$$φ_i(x) = \begin{cases}
1 & \text{if } u_i(x) ≥ \max_{i≠j} u_j(x) \\
0 & \text{otherwise},
\end{cases} \quad (5.1)$$

where $u_i$ are the optimal densities obtained numerically. These contour levels of the functions $φ_i$ almost realize a partition of $S$ with the following issues:

(1) There is a small void space around each triple point, but this void is included in one of the triangles of the mesh, and can be dealt with.
(2) Since we extract the level sets of a function which is either 0 or 1 on the vertices of the triangulation, the contour lines will pass through the middle of the edges of the triangles situated at the border between two phases. This creates some contours which are quite zigzagged and whose length is significantly larger than the optimal total perimeter.

We illustrate these two issues in Figure 6.

Nevertheless, once we have extracted these contours it is possible to make a direct optimization of the total length of the boundaries with the constraint of fixed area of the cells. This optimization is made directly on the triangulated surface. We describe the optimization algorithm below.

**Variables and representation of the partitions.** We denote $(x_i)_{i=1}^h$ a generic family of variable points situated each on an edge of the triangulation $\mathcal{T}$ such that each edge contains exactly one variable point. To these points we associate a family of parameters $(λ_i)_{i=1}^h$ which gives the position of each point $x_i$ on the corresponding edges. We take this global parametric approach since each of these points belongs to at least two cells and we'll need to evaluate its contribution in the gradient of the area and the for all the cells that contain it. Having a global sets of points avoids having to match points between different contours.

Each cell of the partitions is represented by a structure of pairs of edges of triangles of $\mathcal{T}$ which determine, along with the parameters $(λ_i)$, the segments which form the discrete contour of the cell. The pairs of edges is ordered so that the contour is continuous. Contours may have one or more connected components.

**Computation of the perimeters of the cells.** The perimeter of a cell is computed by following the segments forming the contour and incrementally adding their lengths to the total length. If the vertices of the segment are given by $x_i = λ_i v_1 + (1 - λ_i)v_2$ and $x_j = λ_j v_3 + (1 - λ_j)v_4$ then the length of the segment $[x_i, x_j]$ is

$$ℓ([x_i, x_j]) = \|λ_i v_1 + (1 - λ_i)v_2 - λ_j v_3 - (1 - λ_j)v_4\|,$$

expression which is differentiable if the length is not zero. The derivatives with respect to $λ_i$ and $λ_j$ are then added to the gradient vector. Note that for the points which are not vertices of some contour the gradient is zero.

**Computation of the areas of the cells.** In order to compute the area of a cell we use the information given by the functions $φ_i$ defined in (5.1). The function $φ_i$ shows, among other things, what is the position of each triangle in $\mathcal{T}$ with respect to the cell $i$. Indeed, denoting by $T$ a triangle in $\mathcal{T}$, we have the following cases:

(1) All the vertices $v$ of the triangle $T$ satisfy $φ_i(v) = 1$. Then $T$ is completely inside the cell $i$ and we add its area to the total area of the cell.
(2) Two vertices $v_1, v_2$ of $T$ satisfy $φ_i(v_{1,2}) = 1$ and the third satisfies $φ_i(v_3) = 0$. Thus we only add a portion of the area of $T$ to the total area of cell $i$. Note that this value of the area depends linearly of one parameter $λ_k$ and of another parameter $λ_l$. The derivatives of these contributions are added to the vectors containing the gradient of the area of the cell $i$.
(3) Two vertices $v_1, v_2$ of $T$ satisfy $φ_i(v_{1,2}) = 0$ and the third satisfies $φ_i(v_3) = 1$. Again, we only add a portion of the area of $T$ to the total area of cell $i$ which again depends linearly of one parameter $λ_k$ and of another parameter $λ_l$. The derivatives of these contributions are added to the vectors containing the gradient of the area of the cell $i$.
(4) If all the vertices of $T$ satisfy $φ_i(v) = 0$ then the triangle is outside the cell and we move on.

**The empty spaces around triple points.** As we have noted above and seen in Figure 6, around triple points we have some empty spaces determined by three points which belong to the three sides of some of the triangles in $\mathcal{T}$. In each configuration of this type we add a Steiner tree corresponding to the three variable points. Each of the three area regions which are formed are added to the corresponding cell while the perimeter is modified with the length of two adjacent segments in the Steiner tree. See Figure 7 for further details. In order to find the gradient corresponding to the lengths and area changes due to the addition of these Steiner points we use a finite differences approximation.

**Constrained optimization algorithm.** We have the expressions and the gradients of the perimeters and areas of the cells as functions of the parameters $(λ_i)_{i=1}^h$. This allows us to use the algorithm `fmincon` from the Matlab Optimization Toolbox in order to implement the constrained optimization algorithm. We use the interior-point algorithm with a low-memory hessian approximation given by an LBFGS algorithm. The initial values of the parameters $(λ_i)_{i=1}^h$ are all set to 0.5. The algorithm manages to satisfy the constraints at machine precision while minimizing the perimeter and thus smoothing the zigzagged initial contours (like the ones in Figure 6). An example of result may be seen in Figure 8.

It may be the case that some vertices of the contour would "like" to switch to another side. This can be the case if at the end of the optimization one of the parameters $λ_i$ is close to 0 or 1 or a triple point in one of the constructed Steiner trees is on the boundary of the corresponding mesh triangle. In this cases we modify the initial contours taking into the account these results and we restart the optimization procedure. The modification is done in the following way.

(1) If one of the $λ_i$ is equal to 0 or 1 then we add the corresponding point to the adjacent cell and restart the algorithm.
(2) If one of the triple points arrives on the edge of its corresponding mesh triangle then we allow it to move to the adjacent triangle.

After a finite number of switches the configuration stabilizes and a local minimum is found.

We test the presented algorithm on the results obtained in previous sections. In the case of the sphere we obtain the same values found in Table 1. The approximations of the optimal costs for partitions presented in Figure 2 for a torus of radii $R = 1, r = 0.6$ in Table 2.

**Table 2.** Approximation of the optimal costs for minimal partitions of a torus into equal area cells. These partitions are represented in Figure 2

| n | Minimal length |
|---|----------------|
| 2 | 15.07          |
| 3 | 22.61          |
| 4 | 30.15          |
| 5 | 37.25          |
| 6 | 41.93          |
| 7 | 47.12          |
| 8 | 50.77          |
| 9 | 53.37          |
| 10| 56.80          |

## 6. Conclusions

We propose an algorithm for finding numerically the partitions which divide a surface into cells of prescribed areas and minimize the sum of the corresponding perimeters. This algorithm is rigorously justified by a Γ-convergence result which is a generalization of the Modica-Mortola theorem in the case of smooth $(d - 1)$-dimensional manifolds.

In the case of the sphere we are able to recover all the results presented in the article of Cox and Flikkema [9]. The optimal costs of the spherical partitions are precisely evaluated by using the qualitative results in [16], which imply that the boundaries of the cells are arcs of circles. We recover the same optimal costs as the ones presented in [9]. We underline that one of the advantages of this relaxed method is the fact that we do not need to set the polyhedral configuration of the partition a priori. The cells emerge from random density configurations and place themselves in the best positions.

The Γ-convergence method is not limited to the case of the sphere. Once we have triangulated a surface the same algorithm applies. We present a few test cases of more complex surfaces. While the relaxed optimal partitions can easily be obtained, computing the optimal costs is not straightforward since the relaxed costs are not precise enough. In order to be able to compute an approximation of these optimal costs we extract the contours of the optimal densities and we perform a constrained optimization on the triangulated surface.

## References

[1] Giovanni Alberti. Variational models for phase transitions, an approach via gamma-convergence. 1998.

[2] Luigi Ambrosio and Andrea Braides. Functionals defined on partitions in sets of finite perimeter. II. Semicontinuity, relaxation and homogenization. J. Math. Pures Appl. (9), 69(3):307–333, 1990.

[3] Luigi Ambrosio, Nicola Fusco, and Diego Pallara. Functions of bounded variation and free discontinuity problems. Oxford Mathematical Monographs. The Clarendon Press, Oxford University Press, New York, 2000.

[4] Sisto Baldo and Giandomenico Orlandi. Cycles of least mass in a Riemannian manifold, described through the "phase transition" energy of the sections of a line bundle. Math. Z., 225(4):639–655, 1997.

[5] Felix Bernstein. Über die isoperimetrische Eigenschaft des Kreises auf der Kugeloberfläche und in der Ebene. Math. Ann., 60(1):117–136, 1905.

[6] Andrea Braides. Approximation of Free-Discontinuity Problems. Springer, 1998.

[7] Kenneth A. Brakke. The surface evolver. Experiment. Math., 1(2):141–165, 1992.

[8] Giuseppe Buttazzo. Gamma-convergence and its Applications to Some Problems in the Calculus of Variations. School on Homogenization ICTP, Trieste, September 6-17, 1993.

[9] S. J. Cox and E. Flikkema. The minimal perimeter for N confined deformable bubbles of equal area. Electron. J. Combin., 17(1):Research Paper 45, 23, 2010.

[10] Manfredo P. do Carmo. Differential geometry of curves and surfaces. Prentice-Hall, Inc., Englewood Cliffs, N.J., 1976. Translated from the Portuguese.

[11] Max Engelstein. The least-perimeter partition of a sphere into four equal areas. Discrete Comput. Geom., 44(3):645–653, 2010.

[12] Thomas C. Hales. The honeycomb conjecture. Discrete & Computational Geometry, 25(1):1–22, 2001.

[13] Thomas C. Hales. The honeycomb problem on the sphere, 2002.

[14] Antoine Henrot and Michel Pierre. Variation et optimisation de formes, volume 48 of Mathématiques & Applications (Berlin) [Mathematics & Applications]. Springer, Berlin, 2005. Une analyse géométrique. [A geometric analysis].

[15] Joseph D. Masters. The perimeter-minimizing enclosure of two areas in S². Real Anal. Exchange, 22(2):645–654, 1996/97.

[16] Frank Morgan. Soap bubbles in R²and in surfaces. Pacific J. Math., 165(2):347–361, 1994.

[17] Édouard Oudet. Approximation of partitions of least perimeter by Γ-convergence: around Kelvin's conjecture. Exp. Math., 20(3):260–270, 2011.

[18] Theodore Shifrin. Differential geometry - a first course in curves and surfaces.

[19] Liam Stewart. Matlab lbfgs wrapper. http://www.cs.toronto.edu/~liam/software.shtml.

---

(Beniamin Bogosel, Édouard Oudet) Laboratoire Jean Kuntzmann, Université Grenoble Alpes, Bâtiment IMAG, 700 avenue centrale, 38400 Saint Martin d'Hères France

E-mail address, Beniamin Bogosel: beniamin.bogosel@univ-savoie.fr

E-mail address, Édouard Oudet: edouard.oudet@imag.fr