import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import expm


class TimeEvolutionSolver:
    def __init__(self, potential_func, x_min=-5, x_max=5, N=1000, m=1.0, hbar=1.0):
        """Time evolution solver for arbitrary 1D quantum potential."""
        self.N = N
        self.m = m
        self.hbar = hbar
        self.x = np.linspace(x_min, x_max, N)
        self.dx = (x_max - x_min) / (N - 1)
        self.H = self._create_hamiltonian(potential_func)
        self.times = None
        self.psi_t = None

    def _create_hamiltonian(self, potential_func):
        """Create Hamiltonian matrix H = T + V using finite differences."""
        coeff = -(self.hbar ** 2) / (2 * self.m * self.dx ** 2)

        # Build kinetic energy matrix efficiently
        diag_main = np.full(self.N, -2.0, dtype=complex) * coeff
        diag_off = np.ones(self.N - 1, dtype=complex) * coeff

        T = np.diag(diag_main) + np.diag(diag_off, 1) + np.diag(diag_off, -1)
        T[0, -1] = T[-1, 0] = coeff  # Periodic boundary conditions

        return T + np.diag(potential_func(self.x))

    def gaussian_wavepacket(self, x0=0, k0=2, sigma=0.5):
        """Create normalized Gaussian wave packet."""
        psi = np.exp(-(self.x - x0) ** 2 / (4 * sigma ** 2) + 1j * k0 * self.x)
        return psi / np.sqrt(np.trapz(np.abs(psi) ** 2, self.x))

    def time_evolve(self, psi_initial, dt=0.01, n_steps=1000):
        """Evolve wavefunction using matrix exponentiation."""
        if psi_initial.shape[0] != self.N:
            raise ValueError(f"psi_initial length must equal N={self.N}")

        U = expm(-1j * self.H * dt / self.hbar)
        self.times = np.arange(n_steps) * dt
        self.psi_t = np.zeros((n_steps, self.N), dtype=complex)

        psi = psi_initial.copy()
        for i in range(n_steps):
            self.psi_t[i] = psi
            psi = U @ psi
        return self.psi_t

    def animate_evolution(self, save_name=None):
        """Create and optionally save animation of time evolution."""
        y_max = max(np.max(np.abs(self.psi_t.real)), np.max(np.abs(self.psi_t.imag))) * 1.3
        prob_max = np.max(np.abs(self.psi_t) ** 2) * 1.3

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

        # Plot scaled potential if non-zero
        V_diag = np.diag(self.H.real)
        V_plot = V_diag - V_diag.min()
        if V_plot.max() > 1e-10:
            V_plot = V_plot / V_plot.max() * y_max * 0.3
            ax1.plot(self.x, V_plot, 'k--', alpha=0.5, lw=2, label='Potential (scaled)')

        # Initialize plot lines
        line1, = ax1.plot([], [], 'b-', lw=2.5, label='Re(ψ)')
        line2, = ax1.plot([], [], 'r-', lw=2.5, label='Im(ψ)')
        line3, = ax2.plot([], [], 'g-', lw=2.5, label='|ψ|²')

        # Configure axes
        x_range = (self.x.min(), self.x.max())
        for ax in (ax1, ax2):
            ax.set_xlabel('Position x', fontsize=12)
            ax.set_xlim(x_range)
            ax.legend(fontsize=10, loc='upper right')
            ax.grid(True, alpha=0.3)

        ax1.set(ylabel='Wavefunction', ylim=(-y_max, y_max))
        ax1.axhline(0, color='k', lw=0.5, alpha=0.3)
        ax2.set(ylabel='Probability Density', ylim=(0, prob_max))

        def animate(frame):
            psi = self.psi_t[frame]
            line1.set_data(self.x, psi.real)
            line2.set_data(self.x, psi.imag)
            line3.set_data(self.x, np.abs(psi) ** 2)
            ax1.set_title(f'Time Evolution: t = {self.times[frame]:.3f}', fontsize=14)
            return line1, line2, line3

        ani = FuncAnimation(fig, animate, frames=len(self.times), interval=50, blit=True, repeat=True)

        if save_name:
            gifs_dir = 'gifs'
            os.makedirs(gifs_dir, exist_ok=True)
            filepath = os.path.join(gifs_dir, f'{save_name}.gif')
            print(f"Saving animation to '{filepath}'...")
            try:
                ani.save(filepath, writer='pillow', fps=20, dpi=80)
                print(f"✓ Saved as '{filepath}'")
            except Exception as e:
                print(f"✗ Error: {e}")

        plt.show()
        return ani


# Potential functions
def free_potential(x):
    return np.zeros_like(x)

def harmonic_potential(x, omega=1):
    return 0.5 * omega ** 2 * x ** 2

def step_potential(x, height=10, position=0):
    return np.where(x > position, height, 0.0)

def double_well_potential(x, a=0.1, b=1):
    return a * x ** 4 - b * x ** 2


if __name__ == "__main__":
    print("=" * 60)
    print("Quantum Wave Packet Time Evolution")
    print("=" * 60)

    examples = [
        ("Free Particle", free_potential, (-10, 10), (-3, 1.5, 1.0), 'free_particle', 300),
        ("Harmonic Oscillator", harmonic_potential, (-6, 6), (-2, 0, 0.5), 'harmonic_oscillator_time', 400),
        ("Step Potential", lambda x: step_potential(x, 5, 0), (-10, 10), (-4, 2, 1.0), 'quantum_tunneling', 400),
    ]

    for name, potential, (x_min, x_max), (x0, k0, sigma), filename, steps in examples:
        print(f"\n{name}")
        solver = TimeEvolutionSolver(potential, x_min=x_min, x_max=x_max, N=500)
        psi0 = solver.gaussian_wavepacket(x0=x0, k0=k0, sigma=sigma)
        solver.time_evolve(psi0, dt=0.05, n_steps=steps)
        solver.animate_evolution(save_name=filename)

    print("\n" + "=" * 60)
    print("Complete! Check 'gifs/' folder for all animations.")
    print("=" * 60)