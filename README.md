# 2D Heat Equation Solver (Explicit & Implicit Methods)

## Overview

This project solves the two-dimensional heat equation on the unit square using both explicit and implicit finite-difference methods. The objective is to evaluate numerical accuracy, runtime performance, and scalability across multiple computational strategies:

- Serial CPU implementation  
- OpenMP parallelization  
- SIMD vectorization  
- GPU acceleration (Metal / CUDA)  
- Implicit solver using Conjugate Gradient (CG)

Accuracy is quantified using the \( L^2 \) error norm against the analytical solution, and performance is evaluated via wall-clock runtime measurements.

---

## Governing PDE

We solve the heat equation

$$
u_t = \alpha (u_{xx} + u_{yy})
$$

with thermal diffusivity

$$
\alpha = 1
$$

The domain is

$$
(x,y) \in [0,1] \times [0,1], \qquad t \in [0,0.1]
$$

---

## Discretization

Let

- \( n \): x-index  
- \( v \): y-index  
- \( m \): time index  

with the approximation

$$
u(n,v,m) \approx u(x_n, y_v, t_m)
$$

Uniform grid spacing is used:

$$
\Delta x = \Delta y = \frac{1}{512}
$$

---

## Explicit Stability Condition

For the explicit FTCS scheme, stability requires

$$
\alpha \left(
\frac{\Delta t}{(\Delta x)^2}
+
\frac{\Delta t}{(\Delta y)^2}
\right)
\le \frac{1}{2}
$$

The chosen timestep satisfies this condition.

---

## Explicit Finite Difference Scheme (FTCS)

Using forward Euler in time and second-order central differences in space:

$$ \frac{u(n,v,m+1) - u(n,v,m)}{\Delta t}
= u_{xx}(n,v,m) + u_{yy}(n,v,m) $$

where

$$
u_{xx}(n,v,m)
\approx
\frac{u(n+1,v,m) - 2u(n,v,m) + u(n-1,v,m)}{(\Delta x)^2}
$$

$$
u_{yy}(n,v,m)
\approx
\frac{u(n,v+1,m) - 2u(n,v,m) + u(n,v-1,m)}{(\Delta y)^2}
$$

Combining terms gives the update rule:

$$
\begin{aligned}
u(n,v,m+1)
&=
u(n,v,m) \\
&\quad + \Delta t
\left[
\frac{u(n+1,v,m) - 2u(n,v,m) + u(n-1,v,m)}{(\Delta x)^2}
+
\frac{u(n,v+1,m) - 2u(n,v,m) + u(n,v-1,m)}{(\Delta y)^2}
\right]
\end{aligned}
$$

---

## Solution Evolution (Explicit Method)

The animation below shows the time evolution of the temperature field for the explicit FTCS scheme on a \(512 \times 512\) grid with zero Dirichlet boundary conditions.

<p align="center">
  <img src="heat/heat.gif" width="520">
</p>

This visualization provides qualitative validation, demonstrating diffusion-driven smoothing and decay toward the steady state.

---

## Boundary Conditions

Zero Dirichlet boundary conditions are enforced:

\[
u(0,y,t) = u(1,y,t) = 0,
\]

\[
u(x,0,t) = u(x,1,t) = 0.
\]

Boundary values are fixed for all \( t \) and excluded from the implicit linear system.

---

## Implicit Scheme (Backward Euler)

Backward Euler time discretization yields

\[
\frac{u^{m+1} - u^m}{\Delta t} = A u^{m+1},
\]

which can be written as the linear system

\[
(I - \Delta t\, A)\, u^{m+1} = u^m,
\]

where \( A \) is the discrete 2D Laplacian over interior grid points.

Define

\[
\lambda_x = \frac{\Delta t}{(\Delta x)^2}, \qquad
\lambda_y = \frac{\Delta t}{(\Delta y)^2}.
\]

The resulting sparse matrix has:

- Main diagonal:
  \[
  a = 1 + 2\lambda_x + 2\lambda_y
  \]
- Off-diagonals (x-neighbors):
  \[
  -\lambda_x
  \]
- Off-diagonals (y-neighbors):
  \[
  -\lambda_y
  \]

The system is symmetric positive definite and solved using Conjugate Gradient.

---

## Conjugate Gradient (CG) Method

To solve

\[
A x = b,
\]

the CG algorithm proceeds as follows:

**Initialization**
\[
x_0 = 0, \quad r_0 = b - A x_0, \quad p_0 = r_0
\]

**Iteration**
\[
\alpha_k = \frac{r_k^T r_k}{p_k^T A p_k}
\]

\[
x_{k+1} = x_k + \alpha_k p_k
\]

\[
r_{k+1} = r_k - \alpha_k A p_k
\]

\[
\beta_k = \frac{r_{k+1}^T r_{k+1}}{r_k^T r_k}
\]

\[
p_{k+1} = r_{k+1} + \beta_k p_k
\]

Iterations continue until \( \|r_k\|_2 \) falls below a specified tolerance.

---

## Error Measurement (L² Norm)

At final time $T = 0.1$, the error is computed as:

$$ E_{L^2} = \left( \sum_{i,j} \left( u(x_i,y_j,T) - u_h(i,j,T) \right)^2 \, \Delta x \Delta y \right)^{1/2} $$

where:
- $u$ is the analytical solution  
- $u_h$ is the numerical solution  

---

## Project Plan

### Stage 1 — Explicit Solver

Implement the FTCS scheme using:

1. Serial CPU  
2. OpenMP parallelization  
3. SIMD vectorization  
4. GPU acceleration (Metal / CUDA)  

Metrics collected:

- Runtime per timestep  
- Total runtime to \( T = 0.1 \)  
- \( L^2 \) error  

---

### Stage 2 — Implicit Solver

Use Backward Euler time stepping:

\[
(I - \Delta t\, A)\, u^{m+1} = u^m.
\]

Solve using Conjugate Gradient:

1. Serial CG  
2. OpenMP-parallel CG  

Metrics collected:

- Runtime  
- \( L^2 \) error  
- CG iteration counts  
- Scaling with grid size  

---
