import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import eigh
import os


class HarmonicOscillatorSolver:
    """Solver for the 1D quantum harmonic oscillator."""

    def __init__(self, omega=1.0, m=1.0, hbar=1.0, x_max=5.0, N=1000):
        """Initialize harmonic oscillator solver."""
        self.omega = omega
        self.m = m
        self.hbar = hbar
        self.N = N
        self.x = np.linspace(-x_max, x_max, N)
        self.dx = 2 * x_max / (N - 1)
        self.energies = None
        self.wavefunctions = None
        self._create_hamiltonian()

    def _create_hamiltonian(self):
        """Create Hamiltonian matrix H = T + V."""
        coeff = -self.hbar ** 2 / (2.0 * self.m * self.dx ** 2)
        diag = np.full(self.N, -2.0 * coeff)
        off_diag = np.full(self.N - 1, coeff)

        T = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
        T[0, 0] = T[-1, -1] = 1e10
        T[0, 1] = T[-1, -2] = 0

        V = np.diag(0.5 * self.m * self.omega ** 2 * self.x ** 2)
        self.H = T + V

    def solve(self, n_states=5):
        """Solve for eigenvalues and eigenfunctions."""
        eigenvalues, eigenvectors = eigh(self.H)

        mask = eigenvalues > 1e-6
        idx = np.argsort(eigenvalues[mask])[:n_states]

        self.energies = eigenvalues[mask][idx]
        self.wavefunctions = eigenvectors[:, mask][:, idx]

        # Vectorized normalization
        norms = np.sqrt(np.trapz(self.wavefunctions ** 2, self.x, axis=0))
        self.wavefunctions /= norms

        return self.energies, self.wavefunctions

    def analytical_energy(self, n):
        """Analytical energy: E_n = ℏω(n + 1/2)."""
        return self.hbar * self.omega * (n + 0.5)

    def _check_solved(self):
        """Verify that solve() has been called."""
        if self.energies is None:
            raise ValueError("Run solve() first!")

    def _setup_axes(self, axes, xlabel='Position x', fontsize=12):
        """Apply common axis settings."""
        for ax in (axes if isinstance(axes, (list, tuple)) else [axes]):
            ax.set_xlabel(xlabel, fontsize=fontsize)
            ax.set_xlim(self.x.min(), self.x.max())
            ax.legend(fontsize=10, loc='upper right')
            ax.grid(True, alpha=0.3)

    def _save_animation(self, ani, save_name, fps=20, dpi=80):
        """Save animation to GIF."""
        if save_name:
            filepath = f'gifs/{save_name}.gif'
            print(f"Saving '{filepath}'...")
            try:
                ani.save(filepath, writer='pillow', fps=fps, dpi=dpi)
                print(f"✓ Saved as '{filepath}'")
            except Exception as e:
                print(f"✗ Error: {e}")

    def _get_potential_scaled(self, scale_max):
        """Get scaled potential for plotting."""
        V = 0.5 * self.m * self.omega ** 2 * self.x ** 2
        return V / np.max(V) * scale_max * 0.5

    def animate_energy_levels(self, n_states=5, n_frames=100, save_name=None):
        """Create animated visualization cycling through energy levels."""
        self._check_solved()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        V_plot = 0.5 * self.m * self.omega ** 2 * self.x ** 2

        lines_wf, lines_prob, energy_lines = [], [], []

        for i in range(n_states):
            line_wf, = ax1.plot([], [], color=f'C{i}', linewidth=1.5, alpha=0)
            line_prob, = ax2.plot([], [], color=f'C{i}', linewidth=2, alpha=0,
                                  label=f'n={i}, E={self.energies[i]:.3f}')
            e_line = ax1.axhline(y=self.energies[i], color=f'C{i}', linestyle='-',
                                 alpha=0, linewidth=2, label=f'E_{i}={self.energies[i]:.3f}')
            lines_wf.append(line_wf)
            lines_prob.append(line_prob)
            energy_lines.append(e_line)

        ax1.plot(self.x, V_plot, 'k--', label='Potential V(x)', linewidth=2)

        self._setup_axes((ax1, ax2), fontsize=10)
        ax1.set_ylabel('Energy')
        ax1.set_title('Energy Levels and Wavefunctions')
        ax1.set_ylim(0, max(self.energies) * 1.2)

        ax2.set_ylabel('Probability Density |ψ(x)|²')
        ax2.set_title('Probability Densities')

        def animate(frame):
            for i in range(n_states):
                start = int((i / n_states) * n_frames * 0.6)
                end = int(((i + 1) / n_states) * n_frames * 0.6)

                if frame >= start:
                    alpha = min(1.0, (frame - start) / max(1, end - start))
                    psi_shifted = self.wavefunctions[:, i] * 0.5 + self.energies[i]

                    lines_wf[i].set_data(self.x, psi_shifted)
                    lines_wf[i].set_alpha(alpha)
                    lines_prob[i].set_data(self.x, self.wavefunctions[:, i] ** 2)
                    lines_prob[i].set_alpha(alpha)
                    energy_lines[i].set_alpha(alpha * 0.7)

            return lines_wf + lines_prob + energy_lines

        plt.tight_layout()
        ani = FuncAnimation(fig, animate, frames=n_frames, interval=50, blit=True, repeat=True)
        self._save_animation(ani, save_name, dpi=100)
        plt.show()
        return ani

    def _animate_time_evolution(self, times, psi_func, title_func, y_max, prob_max, save_name=None):
        """Generic time evolution animation helper."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        V_scaled = self._get_potential_scaled(y_max)
        ax1.plot(self.x, V_scaled, 'k--', alpha=0.5, linewidth=2, label='Potential (scaled)')

        line1, = ax1.plot([], [], 'b-', linewidth=2.5, label='Re(ψ)')
        line2, = ax1.plot([], [], 'r-', linewidth=2.5, label='Im(ψ)')
        line3, = ax2.plot([], [], 'g-', linewidth=2.5, label='|ψ|²')

        self._setup_axes((ax1, ax2))
        ax1.set_ylabel('Wavefunction', fontsize=12)
        ax1.set_ylim(-y_max, y_max)
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

        ax2.set_ylabel('Probability Density', fontsize=12)
        ax2.set_ylim(0, prob_max)

        def animate(frame):
            t = times[frame]
            psi_t = psi_func(t)
            line1.set_data(self.x, psi_t.real)
            line2.set_data(self.x, psi_t.imag)
            line3.set_data(self.x, np.abs(psi_t) ** 2)
            ax1.set_title(title_func(t), fontsize=14)
            return line1, line2, line3

        plt.tight_layout()
        fig.subplots_adjust(top=0.9)
        ani = FuncAnimation(fig, animate, frames=len(times), interval=50, blit=True, repeat=True)
        self._save_animation(ani, save_name)
        plt.show()
        return ani

    def animate_wavefunction_evolution(self, state_index=0, n_periods=2, n_frames=100, save_name=None):
        """Animate time evolution of a single energy eigenstate."""
        self._check_solved()

        T_period = 2 * np.pi * self.hbar / self.energies[state_index]
        times = np.linspace(0, n_periods * T_period, n_frames)

        y_max = np.max(np.abs(self.wavefunctions[:, state_index])) * 1.2
        prob_max = np.max(self.wavefunctions[:, state_index] ** 2) * 1.2

        def psi_func(t):
            phase = np.exp(-1j * self.energies[state_index] * t / self.hbar)
            return self.wavefunctions[:, state_index] * phase

        def title_func(t):
            return f'Quantum State n={state_index}, E={self.energies[state_index]:.3f}, t={t:.3f}'

        return self._animate_time_evolution(times, psi_func, title_func, y_max, prob_max, save_name)

    def animate_superposition(self, state_indices=None, coefficients=None,
                              n_periods=2, n_frames=100, save_name=None):
        """Animate time evolution of a superposition of energy eigenstates."""
        self._check_solved()

        if state_indices is None:
            state_indices = [0, 1]

        if coefficients is None:
            coefficients = np.ones(len(state_indices))
        coefficients = np.array(coefficients, dtype=float)
        coefficients /= np.linalg.norm(coefficients)

        avg_energy = np.sum(coefficients ** 2 * self.energies[state_indices])
        T_period = 2 * np.pi * self.hbar / avg_energy
        times = np.linspace(0, n_periods * T_period, n_frames)

        def psi_func(t):
            phases = np.exp(-1j * self.energies[state_indices] * t / self.hbar)
            return np.sum(coefficients[:, None] * self.wavefunctions[:, state_indices].T * phases[:, None], axis=0)

        # Pre-calculate limits
        sample_psi = psi_func(0)
        y_max = np.max(np.abs(sample_psi)) * 1.3
        prob_max = np.max(np.abs(sample_psi) ** 2) * 1.3

        states_str = '+'.join([f'ψ_{i}' for i in state_indices])

        def title_func(t):
            return f'Superposition: {states_str}, t={t:.3f}'

        return self._animate_time_evolution(times, psi_func, title_func, y_max, prob_max, save_name)

    def plot_results(self, n_states=4, save_name=None):
        """Plot energy levels and wavefunctions."""
        self._check_solved()

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

        self._setup_axes((ax1, ax2), fontsize=10)
        ax1.set_ylabel('Energy')
        ax1.set_title('Energy Levels and Wavefunctions')
        ax1.set_ylim(0, max(self.energies) * 1.2)

        ax2.set_ylabel('Probability Density |ψ(x)|²')
        ax2.set_title('Probability Densities')

        plt.tight_layout()

        if save_name:
            plt.savefig(f'{save_name}.png', dpi=150, bbox_inches='tight')
            print(f"✓ Static plot saved as '{save_name}.png'")

        plt.show()
        self.print_comparison(n_states)

    def print_comparison(self, n_states=5):
        """Print numerical vs analytical energy comparison."""
        self._check_solved()

        print("\nEnergy Comparison (Numerical vs Analytical):")
        print("=" * 50)
        for i in range(min(n_states, len(self.energies))):
            analytical = self.analytical_energy(i)
            error = abs(self.energies[i] - analytical) / analytical * 100
            print(f"n={i}: {self.energies[i]:.6f} vs {analytical:.6f} (Error: {error:.4f}%)")


if __name__ == "__main__":
    os.makedirs('gifs', exist_ok=True)

    print("=" * 60)
    print("Harmonic Oscillator Quantum Solver")
    print("=" * 60)

    solver = HarmonicOscillatorSolver(omega=1.0, x_max=6.0, N=2000)
    solver.solve(n_states=5)

    animations = [
        ("energy level diagram", lambda: solver.animate_energy_levels(n_states=5, n_frames=120, save_name='harmonic_energy_levels')),
        ("ground state (n=0)", lambda: solver.animate_wavefunction_evolution(state_index=0, save_name='harmonic_ground_state')),
        ("first excited state (n=1)", lambda: solver.animate_wavefunction_evolution(state_index=1, save_name='harmonic_excited_state')),
        ("superposition (n=0 + n=1)", lambda: solver.animate_superposition(state_indices=[0, 1], coefficients=[1, 1], n_periods=3, n_frames=150, save_name='harmonic_superposition')),
    ]

    for i, (name, func) in enumerate(animations):
        print(f"\n{i}. Animating {name}...")
        func()

    print("\n" + "=" * 60)
    print("Complete! Check 'gifs/' folder for animations.")
    print("=" * 60)