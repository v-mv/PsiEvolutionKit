# Quantum Mechanics Schrödinger Equation Solvers

A collection of numerical solvers for the Schrödinger equation in 1D quantum systems with visualization and animation capabilities.

## Overview

This project provides three complete implementations for solving and visualizing quantum mechanical systems:

1. **Infinite Square Well** - Particle confined in a box with infinite potential walls
2. **Harmonic Oscillator** - Quantum harmonic oscillator with parabolic potential
3. **Time Evolution** - Wave packet dynamics in arbitrary potentials

## Features

- ✅ Numerical solution of time-independent Schrödinger equation using finite difference methods
- ✅ Energy eigenvalue and eigenfunction calculations
- ✅ Comparison with analytical solutions
- ✅ Time evolution animations showing wave packet dynamics
- ✅ Quantum superposition visualization
- ✅ GIF export for animations
- ✅ Support for custom potentials

## Requirements

### Dependencies

```bash
numpy
scipy
matplotlib
pillow  # For GIF export
```

### Installation

1. Create a conda environment:
```bash
conda create -n quantum_physics python=3.9
conda activate quantum_physics
```

2. Install required packages:
```bash
conda install numpy scipy matplotlib
```

## Usage

### 1. Infinite Square Well Solver

Solves for a particle confined in a box with walls at x=0 and x=L.

```python
from infinite_well import InfiniteWellSolver

# Create solver
solver = InfiniteWellSolver(L=1.0, m=1.0, hbar=1.0, N=1000)

# Solve for energy eigenstates
energies, wavefunctions = solver.solve(n_states=5)

# Print energy comparison with analytical results
solver.print_comparison(n_states=5)

# Plot static results
solver.plot_results(n_states=4)

# Animate ground state (n=1)
solver.animate_wavefunction_evolution(state_index=0, n_periods=2, 
                                     save_name='ground_state')

# Animate superposition of states
solver.animate_superposition(state_indices=[0, 1], coefficients=[1, 1],
                            save_name='superposition')
```

**Key Results:**
- Energy levels: E_n = (n²π²ℏ²)/(2mL²)
- Wavefunctions: ψ_n(x) = √(2/L) sin(nπx/L)

### 2. Harmonic Oscillator Solver

Solves the quantum harmonic oscillator with potential V(x) = ½mω²x².

```python
from harmonic_oscillator import HarmonicOscillatorSolver

# Create solver
solver = HarmonicOscillatorSolver(omega=1.0, x_max=6.0, N=2000)

# Solve for energy eigenstates
energies, wavefunctions = solver.solve(n_states=5)

# Plot energy levels and wavefunctions
solver.plot_results(n_states=5)

# Animate time evolution
solver.animate_wavefunction_evolution(state_index=0, n_periods=2,
                                     save_name='harmonic_n0')

# Animate superposition (shows oscillating wave packet)
solver.animate_superposition(state_indices=[0, 1], coefficients=[1, 1],
                            n_periods=3, save_name='harmonic_superposition')
```

**Key Results:**
- Energy levels: E_n = ℏω(n + ½)
- Ground state energy: E₀ = ½ℏω (zero-point energy)

### 3. Time Evolution Solver

Simulates real-time wave packet dynamics in arbitrary potentials.

```python
from time_evolution import TimeEvolutionSolver, free_potential, step_potential

# Free particle evolution
solver = TimeEvolutionSolver(free_potential, x_min=-10, x_max=10, N=500)
psi0 = solver.gaussian_wavepacket(x0=-3, k0=1.5, sigma=1.0)
solver.time_evolve(psi0, dt=0.05, n_steps=300)
solver.animate_evolution(save_name='free_particle')

# Quantum tunneling through barrier
solver2 = TimeEvolutionSolver(lambda x: step_potential(x, height=5, position=0),
                              x_min=-10, x_max=10, N=500)
psi0_2 = solver2.gaussian_wavepacket(x0=-4, k0=2, sigma=1.0)
solver2.time_evolve(psi0_2, dt=0.05, n_steps=400)
solver2.animate_evolution(save_name='quantum_tunneling')
```

**Available Potentials:**
- `free_potential(x)` - Free particle
- `harmonic_potential(x, omega)` - Harmonic oscillator
- `step_potential(x, height, position)` - Potential step/barrier
- `double_well_potential(x, a, b)` - Double-well potential

## Running the Examples

Each module can be run directly to see demonstrations:

```bash
# Infinite square well
python infinite_well.py

# Harmonic oscillator
python harmonic_oscillator.py

# Time evolution examples
python time_evolution.py
```

Each script will:
1. Solve for energy eigenstates
2. Display static plots
3. Create and save animations as GIF files
4. Print numerical vs analytical energy comparisons

## Physics Background

### Time-Independent Schrödinger Equation

```
Ĥψ(x) = Eψ(x)
```

Where:
- Ĥ = -ℏ²/(2m) d²/dx² + V(x) is the Hamiltonian operator
- ψ(x) is the wavefunction
- E is the energy eigenvalue

### Time-Dependent Schrödinger Equation

```
iℏ ∂ψ/∂t = Ĥψ(x,t)
```

### Quantum Superposition

When combining multiple energy eigenstates:
```
ψ(x,t) = Σ cₙ ψₙ(x) exp(-iEₙt/ℏ)
```

**Key Insight:** Individual eigenstates have static probability densities, but superpositions show dynamic behavior due to quantum interference!

## Numerical Methods

### Finite Difference Approximation

The second derivative is approximated using a three-point stencil:
```
d²ψ/dx² ≈ [ψ(x+dx) - 2ψ(x) + ψ(x-dx)] / dx²
```

This converts the differential equation into a matrix eigenvalue problem:
```
T·ψ = E·ψ
```

### Time Evolution

Uses matrix exponentiation for time propagation:
```
ψ(t + dt) = exp(-iĤdt/ℏ) ψ(t)
```

## Output Files

Each script generates GIF animations:

**Infinite Well:**
- [`infinite_well_ground_state.gif`](/Notebook/infinite_well_ground_state.gif)
- [`infinite_well_second_state.gif`](/Notebook/infinite_well_second_state.gif)
- [`infinite_well_superposition.gif`](/Notebook/infinite_well_superposition.gif)

**Harmonic Oscillator:**
- [`harmonic_ground_state.gif`](/Notebook/harmonic_ground_state.gif)
- [`harmonic_excited_state.gif`](/Notebook/harmonic_excited_state.gif)
- [`harmonic_superposition.gif`](/Notebook/harmonic_superposition.gif)

**Time Evolution:**
- `free_particle.gif`
- `harmonic_oscillator.gif`
- `quantum_tunneling.gif`

## Understanding the Visualizations

### Single Eigenstate Animations
- **Blue line (Re(ψ))**: Real part of wavefunction - oscillates with phase
- **Red line (Im(ψ))**: Imaginary part of wavefunction - oscillates 90° out of phase
- **Green line (|ψ|²)**: Probability density - **remains static** for eigenstates

### Superposition Animations
- Probability density |ψ|² **oscillates and moves** due to quantum interference
- Demonstrates wave packet "breathing" or "sloshing" motion
- Shows how classical-like motion emerges from quantum superposition

## Project Structure

```
quantum-schrodinger-solvers/
├── infinite_well.py          # Infinite square well solver
├── harmonic_oscillator.py    # Harmonic oscillator solver
├── time_evolution.py         # General time evolution solver
├── README.md                 # This file
└── *.gif                     # Generated animations
```

## Key Concepts Demonstrated

1. **Quantization** - Only discrete energy levels allowed
2. **Zero-point Energy** - Ground state has non-zero energy
3. **Quantum Nodes** - Higher energy states have more nodes
4. **Uncertainty Principle** - Wave packets spread over time
5. **Quantum Tunneling** - Penetration through classically forbidden regions
6. **Superposition** - Linear combination of quantum states
7. **Stationary States** - Energy eigenstates with static probability
8. **Quantum Interference** - Time-dependent behavior in superpositions

## Customization

### Create Custom Potentials

```python
def my_potential(x):
    # Define your potential function
    return np.where(np.abs(x) < 1, 0, 10)  # Finite square well

solver = TimeEvolutionSolver(my_potential, x_min=-5, x_max=5)
```

### Adjust Parameters

```python
# Higher resolution
solver = InfiniteWellSolver(N=2000)

# Different system parameters
solver = HarmonicOscillatorSolver(omega=2.0, m=0.5, hbar=1.0)

# More animation frames
solver.animate_wavefunction_evolution(n_frames=200, n_periods=5)
```

## Validation

All solvers compare numerical results with analytical solutions:
- Infinite well: Typical error < 0.001%
- Harmonic oscillator: Typical error < 0.001%

Errors can be reduced by increasing grid resolution (N).

## Limitations

- 1D systems only
- Assumes time-independent potentials (except time evolution solver)
- Finite grid introduces boundary effects
- Matrix exponentiation can be memory-intensive for large N

## Educational Applications

Perfect for:
- Quantum mechanics courses
- Computational physics projects
- Visualizing abstract quantum concepts
- Understanding numerical methods
- Exploring quantum phenomena interactively

## References

- Griffiths, D.J. (2018). *Introduction to Quantum Mechanics* (3rd ed.)
- Shankar, R. (1994). *Principles of Quantum Mechanics* (2nd ed.)
- Press, W.H. et al. (2007). *Numerical Recipes* (3rd ed.)

## License

This project is open source and available for educational purposes.

## Contributing

Feel free to extend this project with:
- Additional potential functions
- 2D/3D solvers
- Spin dynamics
- Multi-particle systems
- Interactive GUI interfaces

## Contact

For questions or suggestions about this implementation, please open an issue in the repository.

---

**Note:** This is an educational implementation focused on clarity and visualization. For production quantum simulations, consider specialized libraries like QuTiP, PyQuante, or PySCF.# Quantum Mechanics Schrödinger Equation Solvers

A collection of numerical solvers for the Schrödinger equation in 1D quantum systems with visualization and animation capabilities.

## Overview

This project provides three complete implementations for solving and visualizing quantum mechanical systems:

1. **Infinite Square Well** - Particle confined in a box with infinite potential walls
2. **Harmonic Oscillator** - Quantum harmonic oscillator with parabolic potential
3. **Time Evolution** - Wave packet dynamics in arbitrary potentials

## Features

- ✅ Numerical solution of time-independent Schrödinger equation using finite difference methods
- ✅ Energy eigenvalue and eigenfunction calculations
- ✅ Comparison with analytical solutions
- ✅ Time evolution animations showing wave packet dynamics
- ✅ Quantum superposition visualization
- ✅ GIF export for animations
- ✅ Support for custom potentials

## Requirements

### Dependencies

```bash
numpy
scipy
matplotlib
pillow  # For GIF export
```

### Installation

1. Create a conda environment:
```bash
conda create -n quantum_physics python=3.9
conda activate quantum_physics
```

2. Install required packages:
```bash
conda install numpy scipy matplotlib pillow
```

## Usage

### 1. Infinite Square Well Solver

Solves for a particle confined in a box with walls at x=0 and x=L.

```python
from infinite_well import InfiniteWellSolver

# Create solver
solver = InfiniteWellSolver(L=1.0, m=1.0, hbar=1.0, N=1000)

# Solve for energy eigenstates
energies, wavefunctions = solver.solve(n_states=5)

# Print energy comparison with analytical results
solver.print_comparison(n_states=5)

# Plot static results
solver.plot_results(n_states=4)

# Animate ground state (n=1)
solver.animate_wavefunction_evolution(state_index=0, n_periods=2, 
                                     save_name='ground_state')

# Animate superposition of states
solver.animate_superposition(state_indices=[0, 1], coefficients=[1, 1],
                            save_name='superposition')
```

**Key Results:**
- Energy levels: E_n = (n²π²ℏ²)/(2mL²)
- Wavefunctions: ψ_n(x) = √(2/L) sin(nπx/L)

### 2. Harmonic Oscillator Solver

Solves the quantum harmonic oscillator with potential V(x) = ½mω²x².

```python
from harmonic_oscillator import HarmonicOscillatorSolver

# Create solver
solver = HarmonicOscillatorSolver(omega=1.0, x_max=6.0, N=2000)

# Solve for energy eigenstates
energies, wavefunctions = solver.solve(n_states=5)

# Plot energy levels and wavefunctions
solver.plot_results(n_states=5)

# Animate time evolution
solver.animate_wavefunction_evolution(state_index=0, n_periods=2,
                                     save_name='harmonic_n0')

# Animate superposition (shows oscillating wave packet)
solver.animate_superposition(state_indices=[0, 1], coefficients=[1, 1],
                            n_periods=3, save_name='harmonic_superposition')
```

**Key Results:**
- Energy levels: E_n = ℏω(n + ½)
- Ground state energy: E₀ = ½ℏω (zero-point energy)

### 3. Time Evolution Solver

Simulates real-time wave packet dynamics in arbitrary potentials.

```python
from time_evolution import TimeEvolutionSolver, free_potential, step_potential

# Free particle evolution
solver = TimeEvolutionSolver(free_potential, x_min=-10, x_max=10, N=500)
psi0 = solver.gaussian_wavepacket(x0=-3, k0=1.5, sigma=1.0)
solver.time_evolve(psi0, dt=0.05, n_steps=300)
solver.animate_evolution(save_name='free_particle')

# Quantum tunneling through barrier
solver2 = TimeEvolutionSolver(lambda x: step_potential(x, height=5, position=0),
                              x_min=-10, x_max=10, N=500)
psi0_2 = solver2.gaussian_wavepacket(x0=-4, k0=2, sigma=1.0)
solver2.time_evolve(psi0_2, dt=0.05, n_steps=400)
solver2.animate_evolution(save_name='quantum_tunneling')
```

**Available Potentials:**
- `free_potential(x)` - Free particle
- `harmonic_potential(x, omega)` - Harmonic oscillator
- `step_potential(x, height, position)` - Potential step/barrier
- `double_well_potential(x, a, b)` - Double-well potential

## Running the Examples

Each module can be run directly to see demonstrations:

```bash
# Infinite square well
python infinite_well.py

# Harmonic oscillator
python harmonic_oscillator.py

# Time evolution examples
python time_evolution.py
```

Each script will:
1. Solve for energy eigenstates
2. Display static plots
3. Create and save animations as GIF files
4. Print numerical vs analytical energy comparisons

## Physics Background

### Time-Independent Schrödinger Equation

```
Ĥψ(x) = Eψ(x)
```

Where:
- Ĥ = -ℏ²/(2m) d²/dx² + V(x) is the Hamiltonian operator
- ψ(x) is the wavefunction
- E is the energy eigenvalue

### Time-Dependent Schrödinger Equation

```
iℏ ∂ψ/∂t = Ĥψ(x,t)
```

### Quantum Superposition

When combining multiple energy eigenstates:
```
ψ(x,t) = Σ cₙ ψₙ(x) exp(-iEₙt/ℏ)
```

**Key Insight:** Individual eigenstates have static probability densities, but superpositions show dynamic behavior due to quantum interference!

## Numerical Methods

### Finite Difference Approximation

The second derivative is approximated using a three-point stencil:
```
d²ψ/dx² ≈ [ψ(x+dx) - 2ψ(x) + ψ(x-dx)] / dx²
```

This converts the differential equation into a matrix eigenvalue problem:
```
T·ψ = E·ψ
```

### Time Evolution

Uses matrix exponentiation for time propagation:
```
ψ(t + dt) = exp(-iĤdt/ℏ) ψ(t)
```

## Output Files

Each script generates GIF animations:

**Infinite Well:**
- `infinite_well_ground_state.gif`
- `infinite_well_second_state.gif`
- `infinite_well_superposition.gif`

**Harmonic Oscillator:**
- `harmonic_ground_state.gif`
- `harmonic_excited_state.gif`
- `harmonic_superposition.gif`

**Time Evolution:**
- `free_particle.gif`
- `harmonic_oscillator.gif`
- `quantum_tunneling.gif`

## Understanding the Visualizations

### Single Eigenstate Animations
- **Blue line (Re(ψ))**: Real part of wavefunction - oscillates with phase
- **Red line (Im(ψ))**: Imaginary part of wavefunction - oscillates 90° out of phase
- **Green line (|ψ|²)**: Probability density - **remains static** for eigenstates

### Superposition Animations
- Probability density |ψ|² **oscillates and moves** due to quantum interference
- Demonstrates wave packet "breathing" or "sloshing" motion
- Shows how classical-like motion emerges from quantum superposition

## Project Structure

```
quantum-schrodinger-solvers/
├── infinite_well.py          # Infinite square well solver
├── harmonic_oscillator.py    # Harmonic oscillator solver
├── time_evolution.py         # General time evolution solver
├── README.md                 # This file
└── *.gif                     # Generated animations
```

## Key Concepts Demonstrated

1. **Quantization** - Only discrete energy levels allowed
2. **Zero-point Energy** - Ground state has non-zero energy
3. **Quantum Nodes** - Higher energy states have more nodes
4. **Uncertainty Principle** - Wave packets spread over time
5. **Quantum Tunneling** - Penetration through classically forbidden regions
6. **Superposition** - Linear combination of quantum states
7. **Stationary States** - Energy eigenstates with static probability
8. **Quantum Interference** - Time-dependent behavior in superpositions

## Customization

### Create Custom Potentials

```python
def my_potential(x):
    # Define your potential function
    return np.where(np.abs(x) < 1, 0, 10)  # Finite square well

solver = TimeEvolutionSolver(my_potential, x_min=-5, x_max=5)
```

### Adjust Parameters

```python
# Higher resolution
solver = InfiniteWellSolver(N=2000)

# Different system parameters
solver = HarmonicOscillatorSolver(omega=2.0, m=0.5, hbar=1.0)

# More animation frames
solver.animate_wavefunction_evolution(n_frames=200, n_periods=5)
```

## Validation

All solvers compare numerical results with analytical solutions:
- Infinite well: Typical error < 0.001%
- Harmonic oscillator: Typical error < 0.001%

Errors can be reduced by increasing grid resolution (N).

## Limitations

- 1D systems only
- Assumes time-independent potentials (except time evolution solver)
- Finite grid introduces boundary effects
- Matrix exponentiation can be memory-intensive for large N

## Educational Applications

Perfect for:
- Quantum mechanics courses
- Computational physics projects
- Visualizing abstract quantum concepts
- Understanding numerical methods
- Exploring quantum phenomena interactively

## References

- Griffiths, D.J. (2018). *Introduction to Quantum Mechanics* (3rd ed.)
- Shankar, R. (1994). *Principles of Quantum Mechanics* (2nd ed.)
- Press, W.H. et al. (2007). *Numerical Recipes* (3rd ed.)

## License

This project is open source and available for educational purposes.

## Contributing

Feel free to extend this project with:
- Additional potential functions
- 2D/3D solvers
- Spin dynamics
- Multi-particle systems
- Interactive GUI interfaces

## Contact

For questions or suggestions about this implementation, please open an issue in the repository.

---

**Note:** This is an educational implementation focused on clarity and visualization. For production quantum simulations, consider specialized libraries like QuTiP, PyQuante, or PySCF.
