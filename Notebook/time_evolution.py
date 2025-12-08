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
        self.H = self.create_hamiltonian(potential_func)

    def create_hamiltonian(self, potential_func):
        """Create Hamiltonian matrix H = T + V using finite differences."""
        T = np.zeros((self.N, self.N), dtype=complex)
        for i in range(self.N):
            T[i, (i - 1) % self.N] = 1
            T[i, i] = -2
            T[i, (i + 1) % self.N] = 1
        T *= -(self.hbar ** 2) / (2 * self.m * self.dx ** 2)

        V = np.diag(potential_func(self.x))
        return T + V

    def gaussian_wavepacket(self, x0=0, k0=2, sigma=0.5):
        """Create normalized Gaussian wave packet."""
        psi = np.exp(-(self.x - x0) ** 2 / (4 * sigma ** 2)) * np.exp(1j * k0 * self.x)
        norm = np.sqrt(np.trapz(np.abs(psi) ** 2, self.x))
        return psi / norm

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

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot potential if non-zero
        V_plot = np.diag(self.H.real) - np.diag(self.H.real).min()
        if np.max(V_plot) > 1e-10:
            V_plot = V_plot / np.max(V_plot) * y_max * 0.3
            ax1.plot(self.x, V_plot, 'k--', alpha=0.5, linewidth=2, label='Potential (scaled)')

        line1, = ax1.plot([], [], 'b-', linewidth=2.5, label='Re(ψ)')
        line2, = ax1.plot([], [], 'r-', linewidth=2.5, label='Im(ψ)')
        line3, = ax2.plot([], [], 'g-', linewidth=2.5, label='|ψ|²')

        for ax in (ax1, ax2):
            ax.set_xlabel('Position x', fontsize=12)
            ax.legend(fontsize=10, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(self.x.min(), self.x.max())

        ax1.set_ylabel('Wavefunction', fontsize=12)
        ax1.set_ylim(-y_max, y_max)
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

        ax2.set_ylabel('Probability Density', fontsize=12)
        ax2.set_ylim(0, prob_max)

        def animate(frame):
            psi = self.psi_t[frame]
            line1.set_data(self.x, psi.real)
            line2.set_data(self.x, psi.imag)
            line3.set_data(self.x, np.abs(psi) ** 2)
            ax1.set_title(f'Time Evolution: t = {self.times[frame]:.3f}', fontsize=14)
            return line1, line2, line3

        plt.tight_layout()
        ani = FuncAnimation(fig, animate, frames=len(self.times), interval=50, blit=True, repeat=True)

        if save_name:
            print(f"Saving animation to '{save_name}.gif' (this may take 1-2 minutes)...")
            try:
                ani.save(f'{save_name}.gif', writer='pillow', fps=20, dpi=80)
                print(f"✓ Saved as '{save_name}.gif'")
            except Exception as e:
                print(f"✗ Error: {e}\nInstall pillow: conda install pillow")

        plt.show()
        return ani


# Potential functions
def free_potential(x):
    return np.zeros_like(x)


def harmonic_potential(x, omega=1):
    return 0.5 * omega ** 2 * x ** 2


def step_potential(x, height=10, position=0):
    V = np.zeros_like(x)
    V[x > position] = height
    return V


def double_well_potential(x, a=0.1, b=1):
    return a * x ** 4 - b * x ** 2


if __name__ == "__main__":
    print("=" * 60)
    print("Quantum Wave Packet Time Evolution")
    print("=" * 60)

    # Example 1: Free particle
    print("\n1. Free Particle")
    solver = TimeEvolutionSolver(free_potential, x_min=-10, x_max=10, N=500)
    psi0 = solver.gaussian_wavepacket(x0=-3, k0=1.5, sigma=1.0)
    solver.time_evolve(psi0, dt=0.05, n_steps=300)
    solver.animate_evolution(save_name='free_particle')

    # Example 2: Harmonic oscillator
    print("\n2. Harmonic Oscillator")
    solver2 = TimeEvolutionSolver(harmonic_potential, x_min=-6, x_max=6, N=500)
    psi0_2 = solver2.gaussian_wavepacket(x0=-2, k0=0, sigma=0.5)
    solver2.time_evolve(psi0_2, dt=0.05, n_steps=400)
    solver2.animate_evolution(save_name='harmonic_oscillator')

    # Example 3: Step potential (quantum tunneling)
    print("\n3. Step Potential")
    solver3 = TimeEvolutionSolver(lambda x: step_potential(x, height=5, position=0),
                                  x_min=-10, x_max=10, N=500)
    psi0_3 = solver3.gaussian_wavepacket(x0=-4, k0=2, sigma=1.0)
    solver3.time_evolve(psi0_3, dt=0.05, n_steps=400)
    solver3.animate_evolution(save_name='quantum_tunneling')

    print("\n" + "=" * 60)
    print("Complete! Check project folder for GIF files.")
    print("=" * 60)