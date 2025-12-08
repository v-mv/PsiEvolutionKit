import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import eigh


class InfiniteWellSolver:
    def __init__(self, L=1.0, m=1.0, hbar=1.0, N=1000):
        """Initialize the infinite square well solver."""
        self.L = L
        self.m = m
        self.hbar = hbar
        self.N = N
        self.x = np.linspace(0.0, self.L, self.N)
        self.dx = self.L / (self.N - 1)
        self.T = self.create_kinetic_operator()

    def create_kinetic_operator(self):
        """Create kinetic energy operator using finite differences."""
        T = np.zeros((self.N, self.N))
        for i in range(1, self.N - 1):
            T[i, i - 1] = 1.0
            T[i, i] = -2.0
            T[i, i + 1] = 1.0

        T *= -self.hbar ** 2 / (2.0 * self.m * self.dx ** 2)
        T[0, 0] = 1e10
        T[-1, -1] = 1e10
        return T

    def solve(self, n_states=5):
        """Solve for eigenvalues and eigenfunctions."""
        eigenvalues, eigenvectors = eigh(self.T)

        positive_mask = eigenvalues > 1e-6
        positive_eigenvalues = eigenvalues[positive_mask]
        positive_eigenvectors = eigenvectors[:, positive_mask]

        idx = np.argsort(positive_eigenvalues)[:n_states]
        self.energies = positive_eigenvalues[idx]
        self.wavefunctions = positive_eigenvectors[:, idx]

        for i in range(n_states):
            norm = np.sqrt(np.trapz(self.wavefunctions[:, i] ** 2, self.x))
            if norm > 1e-10:
                self.wavefunctions[:, i] /= norm

        return self.energies, self.wavefunctions

    def analytical_energy(self, n):
        """Calculate analytical energy: E_n = (n²π²ℏ²)/(2mL²)"""
        return (n ** 2 * np.pi ** 2 * self.hbar ** 2) / (2 * self.m * self.L ** 2)

    def plot_results(self, n_states=3):
        """Plot wavefunctions and probability densities."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        for i in range(n_states):
            analytical_E = self.analytical_energy(i + 1)
            ax1.plot(self.x, self.wavefunctions[:, i],
                     label=f'n={i + 1}, E={self.energies[i]:.3f} (theory: {analytical_E:.3f})')
            ax2.plot(self.x, self.wavefunctions[:, i] ** 2, label=f'n={i + 1}')

        for ax in (ax1, ax2):
            ax.set_xlabel('Position x')
            ax.legend()
            ax.grid(True, alpha=0.3)

        ax1.set_ylabel('Wavefunction ψ(x)')
        ax1.set_title('Wavefunctions')
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        ax2.set_ylabel('Probability Density |ψ(x)|²')
        ax2.set_title('Probability Densities')

        plt.tight_layout()
        plt.show()

    def print_comparison(self, n_states=5):
        """Print numerical vs analytical energy comparison."""
        print("\nEnergy Comparison (Numerical vs Analytical):")
        print("=" * 50)
        for i in range(min(n_states, len(self.energies))):
            analytical = self.analytical_energy(i + 1)
            error = abs(self.energies[i] - analytical) / analytical * 100
            print(f"n={i + 1}: {self.energies[i]:.6f} vs {analytical:.6f} "
                  f"(Error: {error:.4f}%)")

    def animate_wavefunction_evolution(self, state_index=0, n_periods=2,
                                       n_frames=100, save_name=None):
        """Animate time evolution of a single energy eigenstate."""
        if not hasattr(self, 'energies'):
            raise ValueError("Run solve() first!")

        T_period = 2 * np.pi / (self.energies[state_index] / self.hbar)
        times = np.linspace(0, n_periods * T_period, n_frames)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        ax1.axvline(x=0, color='k', linestyle='-', linewidth=3, alpha=0.8, label='Wall')
        ax1.axvline(x=self.L, color='k', linestyle='-', linewidth=3, alpha=0.8)

        line1, = ax1.plot([], [], 'b-', linewidth=2.5, label='Re(ψ)')
        line2, = ax1.plot([], [], 'r-', linewidth=2.5, label='Im(ψ)')
        line3, = ax2.plot([], [], 'g-', linewidth=2.5, label='|ψ|²')

        for ax in (ax1, ax2):
            ax.set_xlabel('Position x', fontsize=12)
            ax.set_xlim(0, self.L)
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

            ax1.set_title(f'Quantum State n={state_index + 1}, E={self.energies[state_index]:.3f}, t={t:.3f}',
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

        ax1.axvline(x=0, color='k', linestyle='-', linewidth=3, alpha=0.8, label='Wall')
        ax1.axvline(x=self.L, color='k', linestyle='-', linewidth=3, alpha=0.8)

        line1, = ax1.plot([], [], 'b-', linewidth=2.5, label='Re(ψ)')
        line2, = ax1.plot([], [], 'r-', linewidth=2.5, label='Im(ψ)')
        line3, = ax2.plot([], [], 'g-', linewidth=2.5, label='|ψ|²')

        for ax in (ax1, ax2):
            ax.set_xlabel('Position x', fontsize=12)
            ax.set_xlim(0, self.L)
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

            states_str = '+'.join([f'ψ_{i + 1}' for i in state_indices])
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
    print("Infinite Square Well Solver")
    print("=" * 60)

    print("\nSolving for energy eigenstates...")
    solver = InfiniteWellSolver(L=1.0, m=1.0, hbar=1.0, N=1000)
    energies, wavefunctions = solver.solve(n_states=5)

    solver.print_comparison(n_states=5)

    #Print static plot
    print("\nShowing static plot...")
    solver.plot_results(n_states=4)

    #Animated plots
    print("\n1. Animating ground state (n=1)...")
    solver.animate_wavefunction_evolution(state_index=0, n_periods=2, n_frames=100,
                                          save_name='infinite_well_ground_state')

    print("\n2. Animating second state (n=2)...")
    solver.animate_wavefunction_evolution(state_index=1, n_periods=2, n_frames=100,
                                          save_name='infinite_well_second_state')

    print("\n3. Animating superposition (n=1 + n=2)...")
    solver.animate_superposition(state_indices=[0, 1], coefficients=[1, 1],
                                 n_periods=3, n_frames=150,
                                 save_name='infinite_well_superposition')

    print("\n" + "=" * 60)
    print("Complete! Check project folder for GIF files.")
    print("=" * 60)