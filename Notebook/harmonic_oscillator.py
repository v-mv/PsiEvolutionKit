import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import eigh


class HarmonicOscillatorSolver:
    def __init__(self, omega=1.0, m=1.0, hbar=1.0, x_max=5.0, N=1000):
        """Initialize harmonic oscillator solver."""
        self.omega = omega
        self.m = m
        self.hbar = hbar
        self.N = N
        self.x = np.linspace(-x_max, x_max, N)
        self.dx = 2 * x_max / (N - 1)
        self.H = self.create_hamiltonian()

    def create_hamiltonian(self):
        """Create Hamiltonian matrix H = T + V."""
        T = np.zeros((self.N, self.N))
        for i in range(1, self.N - 1):
            T[i, i - 1] = 1.0
            T[i, i] = -2.0
            T[i, i + 1] = 1.0

        T *= -self.hbar ** 2 / (2.0 * self.m * self.dx ** 2)
        T[0, 0] = 1e10
        T[-1, -1] = 1e10

        V = np.diag(0.5 * self.m * self.omega ** 2 * self.x ** 2)
        return T + V

    def solve(self, n_states=5):
        """Solve for eigenvalues and eigenfunctions."""
        eigenvalues, eigenvectors = eigh(self.H)

        positive_mask = eigenvalues > 1e-6
        positive_eigenvalues = eigenvalues[positive_mask]
        positive_eigenvectors = eigenvectors[:, positive_mask]

        idx = np.argsort(positive_eigenvalues)[:n_states]
        self.energies = positive_eigenvalues[idx]
        self.wavefunctions = positive_eigenvectors[:, idx]

        for i in range(n_states):
            norm = np.sqrt(np.trapz(self.wavefunctions[:, i] ** 2, self.x))
            self.wavefunctions[:, i] /= norm
        return self.energies, self.wavefunctions

    def analytical_energy(self, n):
        """Analytical energy: E_n = ℏω(n + 1/2)."""
        return self.hbar * self.omega * (n + 0.5)

    def plot_results(self, n_states=4):
        """Plot energy levels and wavefunctions."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        V_plot = 0.5 * self.m * self.omega ** 2 * self.x ** 2
        ax1.plot(self.x, V_plot, 'k--', label='Potential V(x)', linewidth=2)

        for i in range(n_states):
            ax1.axhline(y=self.energies[i], color=f'C{i}', linestyle='-', alpha=0.7,
                        label=f'E_{i}={self.energies[i]:.3f}')
            psi_shifted = self.wavefunctions[:, i] * 0.5 + self.energies[i]
            ax1.plot(self.x, psi_shifted, color=f'C{i}', linewidth=1.5)
            ax2.plot(self.x, self.wavefunctions[:, i] ** 2,
                     label=f'n={i}, E={self.energies[i]:.3f}')

        ax1.set_xlabel('Position x')
        ax1.set_ylabel('Energy')
        ax1.set_title('Energy Levels and Wavefunctions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, max(self.energies) * 1.2)

        ax2.set_xlabel('Position x')
        ax2.set_ylabel('Probability Density |ψ(x)|²')
        ax2.set_title('Probability Densities')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        self.print_comparison(n_states)

    def print_comparison(self, n_states):
        """Print numerical vs analytical energy comparison."""
        print("\nEnergy Comparison (Numerical vs Analytical):")
        print("=" * 50)
        for i in range(n_states):
            analytical = self.analytical_energy(i)
            error = abs(self.energies[i] - analytical) / analytical * 100
            print(f"n={i}: {self.energies[i]:.6f} vs {analytical:.6f} "
                  f"(Error: {error:.4f}%)")

    def animate_wavefunction_evolution(self, state_index=0, n_periods=2,
                                       n_frames=100, save_name=None):
        """Animate time evolution of a single energy eigenstate."""
        if not hasattr(self, 'energies'):
            raise ValueError("Run solve() first!")

        T_period = 2 * np.pi / (self.energies[state_index] / self.hbar)
        times = np.linspace(0, n_periods * T_period, n_frames)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        V_plot = 0.5 * self.m * self.omega ** 2 * self.x ** 2
        V_scaled = V_plot / np.max(V_plot) * np.max(np.abs(self.wavefunctions[:, state_index])) * 0.5
        ax1.plot(self.x, V_scaled, 'k--', alpha=0.5, linewidth=2, label='Potential (scaled)')

        line1, = ax1.plot([], [], 'b-', linewidth=2.5, label='Re(ψ)')
        line2, = ax1.plot([], [], 'r-', linewidth=2.5, label='Im(ψ)')
        line3, = ax2.plot([], [], 'g-', linewidth=2.5, label='|ψ|²')

        for ax in (ax1, ax2):
            ax.set_xlabel('Position x', fontsize=12)
            ax.set_xlim(self.x.min(), self.x.max())
            ax.legend(fontsize=10, loc='upper right')
            ax.grid(True, alpha=0.3)

        ax1.set_ylabel('Wavefunction', fontsize=12)
        y_max = np.max(np.abs(self.wavefunctions[:, state_index])) * 1.2
        ax1.set_ylim(-y_max, y_max)
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

        ax2.set_ylabel('Probability Density', fontsize=12)
        prob_max = np.max(self.wavefunctions[:, state_index] ** 2) * 1.2
        ax2.set_ylim(0, prob_max)

        def animate(frame):
            t = times[frame]
            phase = np.exp(-1j * self.energies[state_index] * t / self.hbar)
            psi_t = self.wavefunctions[:, state_index] * phase

            line1.set_data(self.x, psi_t.real)
            line2.set_data(self.x, psi_t.imag)
            line3.set_data(self.x, np.abs(psi_t) ** 2)

            ax1.set_title(f'Quantum State n={state_index}, E={self.energies[state_index]:.3f}, t={t:.3f}',
                          fontsize=14)
            return line1, line2, line3

        plt.tight_layout()
        ani = FuncAnimation(fig, animate, frames=n_frames, interval=50, blit=True, repeat=True)

        if save_name:
            print(f"Saving '{save_name}.gif'...")
            try:
                ani.save(f'{save_name}.gif', writer='pillow', fps=20, dpi=80)
                print(f"✓ Saved as '{save_name}.gif'")
            except Exception as e:
                print(f"✗ Error: {e}")

        plt.show()
        return ani

    def animate_superposition(self, state_indices=[0, 1], coefficients=None,
                              n_periods=2, n_frames=100, save_name=None):
        """Animate time evolution of a superposition of energy eigenstates."""
        if not hasattr(self, 'energies'):
            raise ValueError("Run solve() first!")

        if coefficients is None:
            coefficients = np.ones(len(state_indices)) / np.sqrt(len(state_indices))

        coefficients = np.array(coefficients, dtype=float)
        coefficients /= np.linalg.norm(coefficients)

        avg_energy = np.sum([coefficients[i] ** 2 * self.energies[state_indices[i]]
                             for i in range(len(state_indices))])
        T_period = 2 * np.pi / (avg_energy / self.hbar)
        times = np.linspace(0, n_periods * T_period, n_frames)

        # Pre-calculate scaling
        all_psi_real = []
        all_psi_imag = []
        all_prob = []

        for t in times:
            psi_t = np.zeros(self.N, dtype=complex)
            for i, state_idx in enumerate(state_indices):
                phase = np.exp(-1j * self.energies[state_idx] * t / self.hbar)
                psi_t += coefficients[i] * self.wavefunctions[:, state_idx] * phase
            all_psi_real.append(np.abs(psi_t.real))
            all_psi_imag.append(np.abs(psi_t.imag))
            all_prob.append(np.abs(psi_t) ** 2)

        y_max = max(np.max(all_psi_real), np.max(all_psi_imag)) * 1.3
        prob_max = np.max(all_prob) * 1.3

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        V_plot = 0.5 * self.m * self.omega ** 2 * self.x ** 2
        V_scaled = V_plot / np.max(V_plot) * y_max * 0.5
        ax1.plot(self.x, V_scaled, 'k--', alpha=0.5, linewidth=2, label='Potential (scaled)')

        line1, = ax1.plot([], [], 'b-', linewidth=2.5, label='Re(ψ)')
        line2, = ax1.plot([], [], 'r-', linewidth=2.5, label='Im(ψ)')
        line3, = ax2.plot([], [], 'g-', linewidth=2.5, label='|ψ|²')

        for ax in (ax1, ax2):
            ax.set_xlabel('Position x', fontsize=12)
            ax.set_xlim(self.x.min(), self.x.max())
            ax.legend(fontsize=10, loc='upper right')
            ax.grid(True, alpha=0.3)

        ax1.set_ylabel('Wavefunction', fontsize=12)
        ax1.set_ylim(-y_max, y_max)
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

        ax2.set_ylabel('Probability Density', fontsize=12)
        ax2.set_ylim(0, prob_max)

        def animate(frame):
            t = times[frame]
            psi_t = np.zeros(self.N, dtype=complex)

            for i, state_idx in enumerate(state_indices):
                phase = np.exp(-1j * self.energies[state_idx] * t / self.hbar)
                psi_t += coefficients[i] * self.wavefunctions[:, state_idx] * phase

            line1.set_data(self.x, psi_t.real)
            line2.set_data(self.x, psi_t.imag)
            line3.set_data(self.x, np.abs(psi_t) ** 2)

            states_str = '+'.join([f'ψ_{i}' for i in state_indices])
            ax1.set_title(f'Superposition: {states_str}, t={t:.3f}', fontsize=14)
            return line1, line2, line3

        plt.tight_layout()
        ani = FuncAnimation(fig, animate, frames=n_frames, interval=50, blit=True, repeat=True)

        if save_name:
            print(f"Saving '{save_name}.gif'...")
            try:
                ani.save(f'{save_name}.gif', writer='pillow', fps=20, dpi=80)
                print(f"✓ Saved as '{save_name}.gif'")
            except Exception as e:
                print(f"✗ Error: {e}")

        plt.show()
        return ani


if __name__ == "__main__":
    print("=" * 60)
    print("Harmonic Oscillator Quantum Solver")
    print("=" * 60)

    # Solve for energy eigenstates
    print("\nSolving for energy eigenstates...")
    solver = HarmonicOscillatorSolver(omega=1.0, x_max=6.0, N=2000)
    energies, wavefunctions = solver.solve(n_states=5)

    # Plot static results
    print("\nShowing static plot...")
    solver.plot_results(n_states=5)

    # Animate ground state
    print("\n1. Animating ground state (n=0)...")
    solver.animate_wavefunction_evolution(state_index=0, n_periods=2, n_frames=100,
                                          save_name='harmonic_ground_state')

    # Animate first excited state
    print("\n2. Animating first excited state (n=1)...")
    solver.animate_wavefunction_evolution(state_index=1, n_periods=2, n_frames=100,
                                          save_name='harmonic_excited_state')

    # Animate superposition
    print("\n3. Animating superposition (n=0 + n=1)...")
    solver.animate_superposition(state_indices=[0, 1], coefficients=[1, 1],
                                 n_periods=3, n_frames=150,
                                 save_name='harmonic_superposition')

    print("\n" + "=" * 60)
    print("Complete! Check project folder for GIF files.")
    print("=" * 60)