# Theory & Background

This document explains the physics concepts and numerical methods behind the Schrödinger equation solvers.

## Table of Contents
- [Physics Background](#physics-background)
- [Numerical Methods](#numerical-methods)
- [Key Concepts](#key-concepts)

---

## Physics Background

### Time-Independent Schrödinger Equation

The fundamental equation for quantum systems:

```
Ĥψ(x) = Eψ(x)
```

**Components:**
- **Ĥ** - Hamiltonian operator: Ĥ = -ℏ²/(2m) d²/dx² + V(x)
- **ψ(x)** - Wavefunction (describes quantum state)
- **E** - Energy eigenvalue
- **V(x)** - Potential energy function

**Physical Interpretation:**
- The wavefunction ψ(x) contains all information about the quantum system
- |ψ(x)|² gives the probability density of finding the particle at position x
- The Hamiltonian is the total energy operator (kinetic + potential)

### Time-Dependent Schrödinger Equation

For time evolution of quantum states:

```
iℏ ∂ψ/∂t = Ĥψ(x,t)
```

**Solution:**
```
ψ(x,t) = ψ(x,0) exp(-iEt/ℏ)
```

For energy eigenstates, the time dependence is just a phase factor.

### Quantum Superposition

One of the most fundamental principles of quantum mechanics:

```
ψ(x,t) = Σ cₙ ψₙ(x) exp(-iEₙt/ℏ)
```

Where:
- **cₙ** - Complex coefficients (must satisfy Σ|cₙ|² = 1)
- **ψₙ(x)** - Energy eigenstates
- **Eₙ** - Energy eigenvalues

**Critical Insight:**
- Individual eigenstates have **static** probability densities: |ψₙ(x,t)|² = |ψₙ(x)|²
- Superpositions show **dynamic** behavior due to interference between different energy components
- This is why motion only appears in superpositions!

---

## Numerical Methods

### Finite Difference Approximation

The core numerical technique for converting differential equations to matrix equations.

#### Second Derivative Approximation

Using Taylor expansion around point x:

```
ψ(x+dx) ≈ ψ(x) + ψ'(x)dx + ½ψ''(x)dx²
ψ(x-dx) ≈ ψ(x) - ψ'(x)dx + ½ψ''(x)dx²
```

Adding these equations:
```
ψ(x+dx) + ψ(x-dx) ≈ 2ψ(x) + ψ''(x)dx²
```

Rearranging:
```
ψ''(x) ≈ [ψ(x+dx) - 2ψ(x) + ψ(x-dx)] / dx²
```

This gives the **three-point stencil** with coefficients: **[+1, -2, +1]**

#### Matrix Representation

For N grid points, this creates a tridiagonal matrix:

```
        [-2  +1   0   0  ...]
        [+1  -2  +1   0  ...]
T = C × [ 0  +1  -2  +1  ...]
        [ 0   0  +1  -2  ...]
        [        ...        ]
```

Where C = -ℏ²/(2m·dx²)

### Eigenvalue Problem

The Schrödinger equation becomes:
```
(T + V)ψ = Eψ
```

This is a standard matrix eigenvalue problem:
```
H·ψ = E·ψ
```

Where:
- **H = T + V** is the Hamiltonian matrix
- **E** are the eigenvalues (energies)
- **ψ** are the eigenvectors (wavefunctions)

We use `scipy.linalg.eigh()` which is optimized for Hermitian matrices.

### Boundary Conditions

Different quantum systems require different boundary conditions:

**Infinite Square Well:**
- Hard walls: ψ(0) = ψ(L) = 0
- Implemented by setting large diagonal values: T[0,0] = T[-1,-1] = 1e10

**Harmonic Oscillator:**
- Wavefunction decays to zero at ±∞
- Same implementation as infinite well with large enough x_max

**Periodic Boundary Conditions (Time Evolution):**
- ψ(x_min) = ψ(x_max)
- Used in time evolution solver to avoid edge effects
- Implemented using modular arithmetic: T[i, (i±1) % N]

### Time Evolution Method

Uses the formal solution of the time-dependent Schrödinger equation:

```
ψ(t + dt) = exp(-iĤdt/ℏ) ψ(t)
```

**Implementation:**
1. Compute the time evolution operator: U = exp(-iĤdt/ℏ)
2. Apply repeatedly: ψ(t+dt) = U·ψ(t)

**Matrix Exponentiation:**
We use `scipy.linalg.expm()` which computes the matrix exponential accurately using Padé approximation.

---

## Key Concepts

### 1. Quantisation

**Principle:** Only discrete energy levels are allowed in bound quantum systems.

**Why:** Boundary conditions constrain the allowed wavelengths, similar to standing waves on a string.

**Evidence in Code:**
- Infinite well: E_n ∝ n²
- Harmonic oscillator: E_n = ℏω(n + ½)

### 2. Zero-Point Energy

**Principle:** The ground state has non-zero energy.

**Why:** Heisenberg uncertainty principle - particle cannot be at rest (would violate Δx·Δp ≥ ℏ/2)

**Evidence in Code:**
- Harmonic oscillator ground state: E₀ = ½ℏω (not zero!)

### 3. Quantum Nodes

**Principle:** Higher energy states have more nodes (points where ψ=0).

**Why:** More nodes mean higher curvature, thus higher kinetic energy.

**Pattern:**
- State n has (n-1) internal nodes
- Observable in wavefunction plots

### 4. Uncertainty Principle

**Principle:** Δx·Δp ≥ ℏ/2

**Why:** Wave-particle duality - localizing in position spreads out momentum space.

**Evidence in Code:**
- Free particle wave packets spread over time
- Initial narrow packet → large momentum uncertainty → spreading

### 5. Quantum Tunneling

**Principle:** Particles can penetrate classically forbidden regions.

**Why:** Wavefunction is non-zero even in regions where E < V(x).

**Evidence in Code:**
- Step potential simulation shows partial transmission even when E < V
- Wavefunction has exponentially decaying tail in barrier

### 6. Stationary States

**Principle:** Energy eigenstates have time-independent probability densities.

**Why:** Time dependence is just a global phase: exp(-iEt/ℏ)

**Mathematical Proof:**
```
|ψₙ(x,t)|² = |ψₙ(x) exp(-iEₙt/ℏ)|²
           = |ψₙ(x)|² · |exp(-iEₙt/ℏ)|²
           = |ψₙ(x)|² · 1
           = |ψₙ(x)|²
```

The phase factor has unit magnitude!

### 7. Quantum Interference

**Principle:** Probability amplitudes (not probabilities) add in superpositions.

**Why:** Quantum mechanics is linear - states add as complex vectors.

**Mathematical Detail:**

For superposition ψ = c₁ψ₁ + c₂ψ₂:

```
|ψ|² = |c₁ψ₁ + c₂ψ₂|²
     = (c₁ψ₁ + c₂ψ₂)*(c₁*ψ₁* + c₂*ψ₂*)
     = |c₁|²|ψ₁|² + |c₂|²|ψ₂|² + 2Re[c₁c₂*ψ₁*ψ₂]
```

The **interference term** 2Re[c₁c₂*ψ₁*ψ₂] is what creates dynamics!

### 8. Classical Limit

**Principle:** Quantum mechanics reduces to classical mechanics for large quantum numbers or macroscopic systems.

**Why:** Interference effects average out; correspondence principle.

**Evidence in Code:**
- High-n states have many oscillations
- Superpositions can create localized wave packets that move classically

---

## System-Specific Details

### Infinite Square Well

**Potential:**
```
V(x) = { 0      for 0 < x < L
       { ∞      for x ≤ 0 or x ≥ L
```

**Analytical Solutions:**
- Energies: E_n = (n²π²ℏ²)/(2mL²)
- Wavefunctions: ψ_n(x) = √(2/L) sin(nπx/L)
- Nodes: n-1 internal nodes

**Physical Meaning:**
Simplest quantum system - particle completely confined.

### Harmonic Oscillator

**Potential:**
```
V(x) = ½mω²x²
```

**Analytical Solutions:**
- Energies: E_n = ℏω(n + ½)
- Wavefunctions: ψ_n(x) = N_n H_n(√(mω/ℏ)x) exp(-mωx²/2ℏ)
  where H_n are Hermite polynomials

**Physical Meaning:**
Models vibrations, springs, molecular bonds.

**Special Property:**
Equally-spaced energy levels - unique among potentials!

### Free Particle

**Potential:**
```
V(x) = 0  (everywhere)
```

**Properties:**
- Continuous energy spectrum (not quantized)
- Wave packets spread due to dispersion
- Group velocity: v_g = ℏk/m

---

## Numerical Accuracy

### Error Sources

1. **Discretization Error:** O(dx²) for finite differences
2. **Boundary Effects:** Finite domain approximates infinite space
3. **Time Step Error:** For time evolution, depends on dt
4. **Numerical Precision:** Floating-point arithmetic limits

### Improving Accuracy

- **Increase N:** More grid points → smaller dx → better approximation
- **Larger domain:** For harmonic oscillator, ensure wavefunction decays at edges
- **Smaller dt:** For time evolution, reduce time step
- **Higher-order methods:** Could use 5-point stencil for O(dx⁴) accuracy

### Convergence Testing

Always verify convergence:
```python
# Test with increasing resolution
for N in [500, 1000, 2000, 4000]:
    solver = InfiniteWellSolver(N=N)
    energies, _ = solver.solve()
    print(f"N={N}: E₀={energies[0]:.8f}")
```

Results should stabilize as N increases.

---

## Further Reading

**Quantum Mechanics:**
- Griffiths - Introduction to Quantum Mechanics (beginner-friendly)
- Sakurai - Modern Quantum Mechanics (advanced)
- Cohen-Tannoudji - Quantum Mechanics (comprehensive)

**Numerical Methods:**
- Press et al. - Numerical Recipes (practical algorithms)
- Trefethen - Spectral Methods in MATLAB (advanced techniques)
- Newman - Computational Physics (physics-focused)

**Quantum Computing:**
- Nielsen & Chuang - Quantum Computation (connects to quantum states)

---

[← Back to README](../README.md) | [Next: Visualisation Guide →](Visualisation.md)