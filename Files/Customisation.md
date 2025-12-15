# Customisation Guide

This guide shows you how to extend and modify the solvers for your own projects.

## Table of Contents
- [Creating Custom Potentials](#creating-custom-potentials)
- [Adjusting Parameters](#adjusting-parameters)
- [Adding New Features](#adding-new-features)
- [Advanced Modifications](#advanced-modifications)

---

## Creating Custom Potentials

### For Time Evolution Solver

The easiest way to explore new quantum systems is by defining custom potential functions.

### Basic Template

```python
def my_potential(x):
    """
    Define your potential function V(x).
    
    Args:
        x: numpy array of position values
    
    Returns:
        numpy array of potential values at each x
    """
    return # your expression here
```

### Example: Finite Square Well

```python
def finite_well(x, depth=10, width=2):
    """
    Finite square well potential.
    
    V(x) = -depth for |x| < width/2
    V(x) = 0      otherwise
    """
    V = np.zeros_like(x)
    V[np.abs(x) < width/2] = -depth
    return V

# Use it
solver = TimeEvolutionSolver(lambda x: finite_well(x, depth=10, width=2),
                             x_min=-10, x_max=10, N=500)
```

### Example: Triangular Barrier

```python
def triangular_barrier(x, height=5, width=2):
    """
    Triangular potential barrier.
    """
    V = np.zeros_like(x)
    mask = np.abs(x) < width/2
    V[mask] = height * (1 - 2*np.abs(x[mask])/width)
    return V
```

### Example: Periodic Potential (Kronig-Penney)

```python
def periodic_potential(x, V0=5, period=2):
    """
    Periodic potential for modeling crystals.
    """
    return V0 * (1 + np.cos(2*np.pi*x/period)) / 2
```

### Example: Coulomb Potential (1D approximation)

```python
def coulomb_1d(x, strength=1, offset=0.5):
    """
    Approximate Coulomb potential in 1D.
    offset prevents singularity at origin.
    """
    return -strength / (np.abs(x) + offset)
```

### Example: Double Barrier (Quantum Well)

```python
def double_barrier(x, height=8, width=1, separation=3):
    """
    Two barriers creating a quantum well.
    """
    V = np.zeros_like(x)
    # Left barrier
    V[(x > -separation/2 - width) & (x < -separation/2)] = height
    # Right barrier
    V[(x > separation/2) & (x < separation/2 + width)] = height
    return V
```

### Example: Morse Potential (Molecular Vibrations)

```python
def morse_potential(x, D=10, a=1, x0=0):
    """
    Morse potential for diatomic molecules.
    
    D: Dissociation energy
    a: Width parameter
    x0: Equilibrium position
    """
    return D * (1 - np.exp(-a*(x - x0)))**2
```

### Example: Pöschl-Teller Potential

```python
def poschl_teller(x, V0=10, a=1):
    """
    Pöschl-Teller potential (exactly solvable).
    """
    return -V0 / np.cosh(x/a)**2
```

---

## Adjusting Parameters

### Grid Resolution

**Trade-off:** Accuracy vs. Computation Time

```python
# Low resolution (fast, less accurate)
solver = InfiniteWellSolver(N=500)

# Medium resolution (default)
solver = InfiniteWellSolver(N=1000)

# High resolution (slow, more accurate)
solver = InfiniteWellSolver(N=2000)

# Very high resolution (publication quality)
solver = InfiniteWellSolver(N=5000)
```

**When to Increase N:**
- Energy errors > 0.1%
- Wavefunctions look jagged
- High quantum numbers (large n)
- Rapidly varying potentials

### Spatial Domain

**Harmonic Oscillator:**
```python
# Small domain (faster but may cut off wavefunction)
solver = HarmonicOscillatorSolver(x_max=4.0)

# Large domain (ensures wavefunction decays properly)
solver = HarmonicOscillatorSolver(x_max=10.0)
```

**Rule of Thumb:**
- Wavefunction should be < 0.001 at boundaries
- Check: `print(np.max(np.abs(wavefunctions[:, -1])))` at edges

**Time Evolution:**
```python
# Symmetric domain
solver = TimeEvolutionSolver(potential, x_min=-10, x_max=10)

# Asymmetric domain (if needed)
solver = TimeEvolutionSolver(potential, x_min=-5, x_max=15)
```

### Physical Parameters

**Particle Mass:**
```python
# Electron-like (light particle)
solver = InfiniteWellSolver(m=1.0)

# Proton-like (heavy particle)
solver = InfiniteWellSolver(m=1836.0)  # ~1836 times electron mass

# Effect: Heavier particles → lower energies, less spreading
```

**Planck's Constant:**
```python
# Standard units (ℏ = 1)
solver = InfiniteWellSolver(hbar=1.0)

# Physical units (ℏ = 1.054571817×10⁻³⁴ J·s)
solver = InfiniteWellSolver(hbar=1.054571817e-34)
```

**Oscillator Frequency:**
```python
# Slow oscillation
solver = HarmonicOscillatorSolver(omega=0.5)

# Fast oscillation
solver = HarmonicOscillatorSolver(omega=2.0)

# Effect: Higher ω → higher energies, narrower wavefunctions
```

### Time Evolution Parameters

**Time Step:**
```python
# Larger time step (faster but may be unstable)
solver.time_evolve(psi0, dt=0.1, n_steps=200)

# Smaller time step (slower but more accurate)
solver.time_evolve(psi0, dt=0.01, n_steps=2000)
```

**Stability Criterion:**
- For explicit methods: dt < dx²·m/(πℏ)
- If simulation explodes → reduce dt

**Number of Steps:**
```python
# Short simulation
solver.time_evolve(psi0, dt=0.05, n_steps=100)  # Total time = 5

# Long simulation
solver.time_evolve(psi0, dt=0.05, n_steps=1000)  # Total time = 50
```

### Animation Parameters

**Frame Count:**
```python
# Quick preview (choppy)
solver.animate_evolution(n_frames=50)

# Standard (smooth)
solver.animate_evolution(n_frames=100)

# Very smooth (slow to render)
solver.animate_evolution(n_frames=300)
```

**GIF Quality:**
```python
# Web-friendly (small file)
solver.animate_evolution(save_name='web', dpi=60)

# Standard quality
solver.animate_evolution(save_name='standard', dpi=80)

# High quality (large file)
solver.animate_evolution(save_name='hq', dpi=120)
```

---

## Adding New Features

### Custom Initial Wave Packets

Beyond Gaussian packets, you can create custom initial states:

```python
class CustomTimeEvolution(TimeEvolutionSolver):
    def triangular_wavepacket(self, x0=0, width=2):
        """Triangular wave packet."""
        psi = np.zeros(self.N, dtype=complex)
        mask = np.abs(self.x - x0) < width/2
        psi[mask] = 1 - 2*np.abs(self.x[mask] - x0)/width
        norm = np.sqrt(np.trapz(np.abs(psi)**2, self.x))
        return psi / norm
    
    def rectangular_wavepacket(self, x0=0, width=2):
        """Rectangular wave packet."""
        psi = np.zeros(self.N, dtype=complex)
        psi[np.abs(self.x - x0) < width/2] = 1.0
        norm = np.sqrt(np.trapz(np.abs(psi)**2, self.x))
        return psi / norm
```

### Multiple Superposition States

Combine more than two states:

```python
# Three-state superposition
solver.animate_superposition(
    state_indices=[0, 1, 2],
    coefficients=[1, 1, 1],  # Equal weights
    n_periods=5,
    save_name='three_state'
)

# Weighted superposition
solver.animate_superposition(
    state_indices=[0, 1, 2],
    coefficients=[0.8, 0.5, 0.3],  # Different weights
    n_periods=5,
    save_name='weighted'
)
```

### Expectation Values

Add methods to calculate observable quantities:

```python
class ExtendedSolver(InfiniteWellSolver):
    def expectation_position(self, state_index):
        """Calculate <x> for a state."""
        psi = self.wavefunctions[:, state_index]
        return np.trapz(self.x * np.abs(psi)**2, self.x)
    
    def expectation_momentum(self, state_index):
        """Calculate <p> (approximately)."""
        psi = self.wavefunctions[:, state_index]
        dpsi_dx = np.gradient(psi, self.dx)
        # <p> = -iℏ ∫ ψ* dψ/dx dx
        integrand = -1j * self.hbar * np.conj(psi) * dpsi_dx
        return np.trapz(integrand, self.x).real
    
    def uncertainty_position(self, state_index):
        """Calculate Δx."""
        psi = self.wavefunctions[:, state_index]
        x_avg = self.expectation_position(state_index)
        x2_avg = np.trapz(self.x**2 * np.abs(psi)**2, self.x)
        return np.sqrt(x2_avg - x_avg**2)
```

### Energy Level Diagram

Create a custom visualization:

```python
def plot_energy_diagram(solver, n_states=5):
    """Plot energy level diagram with transitions."""
    fig, ax = plt.subplots(figsize=(8, 10))
    
    for i in range(n_states):
        # Draw energy level
        ax.hlines(solver.energies[i], 0, 1, colors='blue', linewidth=2)
        ax.text(1.05, solver.energies[i], f'n={i+1}, E={solver.energies[i]:.3f}',
                va='center')
        
        # Draw transitions
        for j in range(i):
            # Selection rules can be implemented here
            ax.arrow(0.5, solver.energies[j], 0, 
                    solver.energies[i] - solver.energies[j],
                    head_width=0.05, head_length=0.1*solver.energies[0],
                    fc='red', ec='red', alpha=0.3)
    
    ax.set_xlim(-0.2, 1.5)
    ax.set_ylabel('Energy')
    ax.set_title('Energy Level Diagram')
    ax.set_xticks([])
    plt.show()
```

### Save Wavefunction Data

Export data for further analysis:

```python
def save_wavefunctions(solver, filename='wavefunctions.npz'):
    """Save wavefunctions and energies to file."""
    np.savez(filename,
             x=solver.x,
             energies=solver.energies,
             wavefunctions=solver.wavefunctions)
    print(f"Data saved to {filename}")

def load_wavefunctions(filename='wavefunctions.npz'):
    """Load previously saved data."""
    data = np.load(filename)
    return data['x'], data['energies'], data['wavefunctions']
```

---

## Advanced Modifications

### 2D Extension

Extend to two dimensions:

```python
class Solver2D:
    def __init__(self, Lx=1.0, Ly=1.0, Nx=100, Ny=100):
        """2D quantum system."""
        self.Nx = Nx
        self.Ny = Ny
        self.x = np.linspace(0, Lx, Nx)
        self.y = np.linspace(0, Ly, Ny)
        self.dx = Lx / (Nx - 1)
        self.dy = Ly / (Ny - 1)
        
        # Create 2D grid
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
    def create_hamiltonian_2d(self, potential_func):
        """Build 2D Hamiltonian using Kronecker products."""
        # 1D operators
        T_x = self.kinetic_1d(self.Nx, self.dx)
        T_y = self.kinetic_1d(self.Ny, self.dy)
        I_x = np.eye(self.Nx)
        I_y = np.eye(self.Ny)
        
        # 2D kinetic energy: T_x ⊗ I_y + I_x ⊗ T_y
        T_2d = np.kron(T_x, I_y) + np.kron(I_x, T_y)
        
        # Potential energy
        V_values = potential_func(self.X, self.Y).flatten()
        V_2d = np.diag(V_values)
        
        return T_2d + V_2d
```

### Sparse Matrix Implementation

For large systems, use sparse matrices:

```python
from scipy.sparse import diags, eye
from scipy.sparse.linalg import eigsh

class SparseSolver(InfiniteWellSolver):
    def create_kinetic_operator(self):
        """Use sparse matrix for large N."""
        diagonals = [np.ones(self.N-1),
                    -2*np.ones(self.N),
                    np.ones(self.N-1)]
        T = diags(diagonals, [-1, 0, 1], format='csr')
        T *= -self.hbar**2 / (2.0 * self.m * self.dx**2)
        
        # Boundary conditions
        T[0, 0] = 1e10
        T[-1, -1] = 1e10
        return T
    
    def solve(self, n_states=5):
        """Use sparse eigenvalue solver."""
        eigenvalues, eigenvectors = eigsh(self.T, k=n_states, which='SA')
        # 'SA' = Smallest Algebraic (lowest energies)
        self.energies = eigenvalues
        self.wavefunctions = eigenvectors
        # Normalise...
        return self.energies, self.wavefunctions
```

### Higher-Order Finite Differences

Improve accuracy with 5-point stencil:

```python
def create_kinetic_operator_higher_order(self):
    """5-point stencil: O(dx⁴) accuracy."""
    T = np.zeros((self.N, self.N))
    
    for i in range(2, self.N-2):
        T[i, i-2] = -1/12
        T[i, i-1] = 4/3
        T[i, i] = -5/2
        T[i, i+1] = 4/3
        T[i, i+2] = -1/12
    
    T *= -self.hbar**2 / (2.0 * self.m * self.dx**2)
    # Handle boundaries...
    return T
```

### Imaginary Time Evolution

Find ground state by evolving in imaginary time:

```python
def find_ground_state_imaginary_time(solver, psi_initial, dt_imag=0.01, n_steps=1000):
    """
    Evolve in imaginary time to project out ground state.
    ψ(τ) = exp(-Ĥτ/ℏ) ψ(0)
    Exponentially suppresses excited states.
    """
    U = expm(-solver.H * dt_imag / solver.hbar)  # Real exponential
    
    psi = psi_initial.copy()
    for i in range(n_steps):
        psi = U @ psi
        # Renormalise
        norm = np.sqrt(np.trapz(np.abs(psi)**2, solver.x))
        psi /= norm
    
    return psi
```

### Add Spin

Include spin-½ particles:

```python
class SpinorSolver:
    def create_hamiltonian_with_spin(self, potential_func, B_field=0):
        """
        Include Zeeman term for spin in magnetic field.
        ψ = [ψ_up, ψ_down]
        """
        # Spatial Hamiltonian
        H_spatial = self.create_hamiltonian(potential_func)
        
        # Identity for spin
        I_spin = np.eye(2)
        I_space = np.eye(self.N)
        
        # Spatial part: H ⊗ I_spin
        H_total = np.kron(H_spatial, I_spin)
        
        # Zeeman term: I_space ⊗ (μ_B * B * σ_z)
        sigma_z = np.array([[1, 0], [0, -1]])
        H_zeeman = np.kron(I_space, B_field * sigma_z)
        
        return H_total + H_zeeman
```

### Adaptive Time Stepping

Automatically adjust dt for stability:

```python
def adaptive_time_evolve(solver, psi_initial, total_time=10, tol=1e-6):
    """
    Adaptive time stepping based on error estimates.
    """
    t = 0
    dt = 0.01  # Initial guess
    psi = psi_initial.copy()
    times = [t]
    psi_history = [psi.copy()]
    
    while t < total_time:
        # Try step with dt
        U = expm(-1j * solver.H * dt / solver.hbar)
        psi_new = U @ psi
        
        # Try step with dt/2 twice
        U_half = expm(-1j * solver.H * (dt/2) / solver.hbar)
        psi_half = U_half @ U_half @ psi
        
        # Estimate error
        error = np.linalg.norm(psi_new - psi_half)
        
        if error < tol:
            # Accept step
            psi = psi_new
            t += dt
            times.append(t)
            psi_history.append(psi.copy())
            # Increase dt if error is very small
            if error < tol/10:
                dt *= 1.5
        else:
            # Reject step, reduce dt
            dt *= 0.5
    
    return np.array(times), np.array(psi_history)
```

---

## Best Practices

### Code Organization

For larger projects:

```python
# quantum_solvers/
#   __init__.py
#   base.py          # Base solver class
#   infinite_well.py
#   harmonic.py
#   time_evolution.py
#   potentials.py    # Collection of potential functions
#   visualization.py # Plotting utilities
#   analysis.py      # Expectation values, uncertainties
```

### Testing

Always validate against known solutions:

```python
import pytest

def test_infinite_well_energies():
    solver = InfiniteWellSolver(L=1.0, m=1.0, hbar=1.0, N=2000)
    energies, _ = solver.solve(n_states=5)
    
    for n in range(1, 6):
        analytical = (n**2 * np.pi**2) / 2
        numerical = energies[n-1]
        error = abs(numerical - analytical) / analytical
        assert error < 0.0001, f"Energy error too large for n={n}"

def test_normalization():
    solver = InfiniteWellSolver(N=1000)
    _, wavefunctions = solver.solve(n_states=3)
    
    for i in range(3):
        norm = np.trapz(wavefunctions[:, i]**2, solver.x)
        assert abs(norm - 1.0) < 0.001, f"State {i} not normalised"
```

### Documentation

Add docstrings to custom functions:

```python
def my_potential(x, param1, param2):
    """
    Brief description of the potential.
    
    Args:
        x (np.ndarray): Position array
        param1 (float): Description of param1
        param2 (float): Description of param2
    
    Returns:
        np.ndarray: Potential energy values
        
    Example:
        >>> V = my_potential(x, param1=5, param2=2)
    
    Notes:
        Any important physical or numerical considerations.
    """
    pass
```

---

## Project Ideas

### Educational

1. **Interactive Jupyter Notebook:**
   - Sliders for parameters
   - Real-time visualization updates
   - ipywidgets integration

2. **Comparison Tool:**
   - Side-by-side different potentials
   - Parameter sweep visualizations
   - Phase space plots

3. **Quantum Game:**
   - Guide particle through obstacles
   - Visualize probability flow
   - Educational mini-games

### Research

1. **Scattering Problems:**
   - Transmission coefficients vs. energy
   - Resonance phenomena
   - Phase shifts

2. **Periodic Systems:**
   - Band structure calculations
   - Bloch waves
   - Kronig-Penney model

3. **Time-Dependent Potentials:**
   - Driven harmonic oscillator
   - Floquet states
   - Multiphoton processes

---

[← Back to Visualisation](Visualisation.md) | [Return to Readme](../README.md)