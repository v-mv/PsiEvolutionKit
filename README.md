# Quantum Mechanics Schrödinger Equation Solvers

A collection of numerical solvers for the Schrödinger equation in 1D quantum systems with visualisation and animation capabilities.

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
- ✅ Quantum superposition visualisation
- ✅ GIF export for animations
- ✅ Support for custom potentials

## Quick Start

### Installation

```bash
# Create conda environment
conda create -n quantum_physics python=3.9
conda activate quantum_physics

# Install dependencies
conda install numpy scipy matplotlib pillow
```

### Run Examples

```bash
# Infinite square well
python infinite_well.py

# Harmonic oscillator
python harmonic_oscillator.py

# Time evolution examples
python time_evolution.py
```

Each script will generate static plots, animations, and save GIF files.

## Usage Examples

### Infinite Square Well

```python
from infinite_well import InfiniteWellSolver

solver = InfiniteWellSolver(L=1.0, m=1.0, hbar=1.0, N=1000)
energies, wavefunctions = solver.solve(n_states=5)

solver.print_comparison(n_states=5)
solver.plot_results(n_states=4)

# Animate ground state
solver.animate_wavefunction_evolution(state_index=0, save_name='ground_state')

# Animate superposition
solver.animate_superposition(state_indices=[0, 1], coefficients=[1, 1],
                            save_name='superposition')
```

### Harmonic Oscillator

```python
from harmonic_oscillator import HarmonicOscillatorSolver

solver = HarmonicOscillatorSolver(omega=1.0, x_max=6.0, N=2000)
energies, wavefunctions = solver.solve(n_states=5)

solver.plot_results(n_states=5)
solver.animate_superposition(state_indices=[0, 1], coefficients=[1, 1],
                            save_name='harmonic_superposition')
```

### Time Evolution

```python
from time_evolution import TimeEvolutionSolver, free_potential, step_potential

# Free particle
solver = TimeEvolutionSolver(free_potential, x_min=-10, x_max=10, N=500)
psi0 = solver.gaussian_wavepacket(x0=-3, k0=1.5, sigma=1.0)
solver.time_evolve(psi0, dt=0.05, n_steps=300)
solver.animate_evolution(save_name='free_particle')

# Quantum tunneling
solver2 = TimeEvolutionSolver(lambda x: step_potential(x, height=5, position=0),
                              x_min=-10, x_max=10, N=500)
psi0_2 = solver2.gaussian_wavepacket(x0=-4, k0=2, sigma=1.0)
solver2.time_evolve(psi0_2, dt=0.05, n_steps=400)
solver2.animate_evolution(save_name='quantum_tunneling')
```

## Output Files

Each script generates GIF animations in the `gifs/` folder:

**Infinite Well:**
- `infinite_well_energy_levels.gif` - Animated energy level diagram
- `infinite_well_ground_state.gif` - Ground state (n=1) time evolution
- `infinite_well_second_state.gif` - Second state (n=2) time evolution
- `infinite_well_superposition.gif` - Superposition of n=1 and n=2

**Harmonic Oscillator:**
- `harmonic_energy_levels.gif` - Animated energy level diagram
- `harmonic_ground_state.gif` - Ground state (n=0) time evolution
- `harmonic_excited_state.gif` - First excited state (n=1) time evolution
- `harmonic_superposition.gif` - Superposition of n=0 and n=1

**Time Evolution:**
- `free_particle.gif` - Free particle wave packet spreading
- `harmonic_oscillator_time.gif` - Wave packet in harmonic potential
- `quantum_tunneling.gif` - Tunneling through step potential barrier

**Total: 11 animated GIF files** showcasing different quantum phenomena!

## Documentation

- [📖 Theory & Background](Files/Theory.md) - Physics concepts and numerical methods
- [🎨 Visualisation Guide](Files/Visualisation.md) - Complete guide to all animations
- [⚙️ Customisation Guide](Files/Customisation.md) - Extending and modifying the code

## Project Structure

```
Schrodinger-eq/
├── README.md                 # This file
├── Files/
│   ├── Theory.md            # Physics and numerical methods
│   ├── Visualisation.md     # Complete guide to all 11 animations
│   └── Customisation.md     # How to extend the code
├──Notebooks/                 # Jupyter notebooks for interactive exploration
    └── gifs/                     # All generated animations (11 files)
        ├── infinite_well_energy_levels.gif
        ├── infinite_well_ground_state.gif
        ├── infinite_well_second_state.gif
        ├── infinite_well_superposition.gif
        ├── harmonic_energy_levels.gif
        ├── harmonic_ground_state.gif
        ├── harmonic_excited_state.gif
        ├── harmonic_superposition.gif
        ├── free_particle.gif
        ├── harmonic_oscillator_time.gif
        └── quantum_tunneling.gif
    ├── infinite_well.py          # Infinite square well solver
    ├── harmonic_oscillator.py    # Harmonic oscillator solver
    ├── time_evolution.py         # General time evolution solver    
```

## Key Results

**Infinite Square Well:**
- Energy levels: E_n = (n²π²ℏ²)/(2mL²)
- Numerical error: < 0.001%

**Harmonic Oscillator:**
- Energy levels: E_n = ℏω(n + ½)
- Zero-point energy: E₀ = ½ℏω
- Numerical error: < 0.001%

## Requirements

- Python 3.9+
- NumPy
- SciPy
- Matplotlib
- Pillow (for GIF export)

## Educational Applications

Perfect for:
- Quantum mechanics courses
- Computational physics projects
- Visualizing abstract quantum concepts
- Understanding numerical methods
- Exploring quantum phenomena interactively

## Validation

All solvers compare numerical results with analytical solutions where available. Errors can be reduced by increasing grid resolution (N parameter).

## References

- Griffiths, D.J. (2018). *Introduction to Quantum Mechanics* (3rd ed.)
- Shankar, R. (1994). *Principles of Quantum Mechanics* (2nd ed.)
- Press, W.H. et al. (2007). *Numerical Recipes* (3rd ed.)

## License

This project is open source and available for educational purposes.

## Contributing

Feel free to extend this project with additional features. See [CUSTOMISATION.md](Files/Customisation.md) for ideas.

---

**Note:** This is an educational implementation focused on clarity and visualisation. For production quantum simulations, consider specialized libraries like QuTiP, PyQuante, or PySCF.