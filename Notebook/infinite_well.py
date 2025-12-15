import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import eigh
import os


class InfiniteWellSolver:
    """Solver for the 1D infinite square well quantum system."""

    def __init__(self, L=1.0, m=1.0, hbar=1.0, N=1000):
        """Initialize the infinite square well solver."""
        self.L = L
        self.m = m
        self.hbar = hbar
        self.N = N
        self.x = np.linspace(0.0, L, N)
        self.dx = L / (N - 1)
        self.energies = None
        self.wavefunctions = None
        self._create_hamiltonian()

    def _create_hamiltonian(self):
        """Create kinetic energy operator using finite differences."""
        coeff = -self.hbar ** 2 / (2.0 * self.m * self.dx ** 2)
        diag = np.full(self.N, -2.0 * coeff)
        off_diag = np.full(self.N - 1, coeff)

        self.H = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
        # Enforce boundary conditions
        self.H[0, 0] = self.H[-1, -1] = 1e10
        self.H[0, 1] = self.H[-1, -2] = 0

    def solve(self, n_states=5):
        """Solve for eigenvalues and eigenfunctions."""
        eigenvalues, eigenvectors = eigh(self.H)

        # Filter positive eigenvalues and sort
        mask = eigenvalues > 1e-6
        idx = np.argsort(eigenvalues[mask])[:n_states]

        self.energies = eigenvalues[mask][idx]
        self.wavefunctions = eigenvectors[:, mask][:, idx]

        # Normalize wavefunctions
        norms = np.sqrt(np.trapz(self.wavefunctions ** 2, self.x, axis=0))
        self.wavefunctions /= norms

        return self.energies, self.wavefunctions

    def analytical_energy(self, n):
        """Calculate analytical energy: E_n = (n²π²ℏ²)/(2mL²)"""
        return (n ** 2 * np.pi ** 2 * self.hbar ** 2) / (2 * self.m * self.L ** 2)

    def _check_solved(self):
        """Verify that solve() has been called."""
        if self.energies is None:
            raise ValueError("Run solve() first!")

    def _setup_axes(self, axes, xlabel='Position x', fontsize=12):
        """Apply common axis settings."""
        for ax in (axes if isinstance(axes, (list, tuple)) else [axes]):
            ax.set_xlabel(xlabel, fontsize=fontsize)
            ax.set_xlim(0, self.L)
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

    def animate_energy_levels(self, n_states=4, n_frames=100, save_name=None):
        """Create animated visualization cycling through energy levels."""
        self._check_solved()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        lines_wf, lines_prob = [], []

        for i in range(n_states):
            analytical_E = self.analytical_energy(i + 1)
            line_wf, = ax1.plot([], [], color=f'C{i}', linewidth=2, alpha=0,
                                label=f'n={i + 1}, E={self.energies[i]:.3f} (theory: {analytical_E:.3f})')
            line_prob, = ax2.plot([], [], color=f'C{i}', linewidth=2, alpha=0, label=f'n={i + 1}')
            lines_wf.append(line_wf)
            lines_prob.append(line_prob)

        self._setup_axes((ax1, ax2), fontsize=10)
        ax1.set_ylabel('Wavefunction ψ(x)')
        ax1.set_title('Wavefunctions')
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax1.set_ylim(-np.max(np.abs(self.wavefunctions)) * 1.1, np.max(np.abs(self.wavefunctions)) * 1.1)

        ax2.set_ylabel('Probability Density |ψ(x)|²')
        ax2.set_title('Probability Densities')
        ax2.set_ylim(0, np.max(self.wavefunctions ** 2) * 1.1)

        def animate(frame):
            for i in range(n_states):
                start = int((i / n_states) * n_frames * 0.6)
                end = int(((i + 1) / n_states) * n_frames * 0.6)

                if frame >= start:
                    alpha = min(1.0, (frame - start) / max(1, end - start))
                    lines_wf[i].set_data(self.x, self.wavefunctions[:, i])
                    lines_wf[i].set_alpha(alpha)
                    lines_prob[i].set_data(self.x, self.wavefunctions[:, i] ** 2)
                    lines_prob[i].set_alpha(alpha)
            return lines_wf + lines_prob

        plt.tight_layout()
        ani = FuncAnimation(fig, animate, frames=n_frames, interval=50, blit=True, repeat=True)
        self._save_animation(ani, save_name, dpi=100)
        plt.show()
        return ani

    def _animate_time_evolution(self, times, psi_func, title_func, save_name=None):
        """Generic time evolution animation helper."""
        # Pre-calculate for y-limits
        sample_psi = psi_func(0)
        y_max = np.max(np.abs(sample_psi)) * 1.3
        prob_max = np.max(np.abs(sample_psi) ** 2) * 1.3

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        ax1.axvline(x=0, color='k', linestyle='-', linewidth=3, alpha=0.8, label='Wall')
        ax1.axvline(x=self.L, color='k', linestyle='-', linewidth=3, alpha=0.8)

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

        def psi_func(t):
            phase = np.exp(-1j * self.energies[state_index] * t / self.hbar)
            return self.wavefunctions[:, state_index] * phase

        def title_func(t):
            return f'Quantum State n={state_index + 1}, E={self.energies[state_index]:.3f}, t={t:.3f}'

        return self._animate_time_evolution(times, psi_func, title_func, save_name)

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

        states_str = '+'.join([f'ψ_{i + 1}' for i in state_indices])
        def title_func(t):
            return f'Superposition: {states_str}, t={t:.3f}'

        return self._animate_time_evolution(times, psi_func, title_func, save_name)

    def plot_results(self, n_states=3):
        """Plot wavefunctions and probability densities."""
        self._check_solved()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        for i in range(n_states):
            analytical_E = self.analytical_energy(i + 1)
            ax1.plot(self.x, self.wavefunctions[:, i],
                     label=f'n={i + 1}, E={self.energies[i]:.3f} (theory: {analytical_E:.3f})')
            ax2.plot(self.x, self.wavefunctions[:, i] ** 2, label=f'n={i + 1}')

        self._setup_axes((ax1, ax2), fontsize=10)
        ax1.set_ylabel('Wavefunction ψ(x)')
        ax1.set_title('Wavefunctions')
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax2.set_ylabel('Probability Density |ψ(x)|²')
        ax2.set_title('Probability Densities')

        plt.tight_layout()
        plt.show()

    def print_comparison(self, n_states=5):
        """Print numerical vs analytical energy comparison."""
        self._check_solved()

        print("\nEnergy Comparison (Numerical vs Analytical):")
        print("=" * 50)
        for i in range(min(n_states, len(self.energies))):
            analytical = self.analytical_energy(i + 1)
            error = abs(self.energies[i] - analytical) / analytical * 100
            print(f"n={i + 1}: {self.energies[i]:.6f} vs {analytical:.6f} (Error: {error:.4f}%)")


if __name__ == "__main__":
    os.makedirs('gifs', exist_ok=True)

    print("=" * 60)
    print("Infinite Square Well Solver")
    print("=" * 60)

    solver = InfiniteWellSolver(L=1.0, m=1.0, hbar=1.0, N=1000)
    solver.solve(n_states=5)
    solver.print_comparison(n_states=5)

    animations = [
        ("energy level diagram", lambda: solver.animate_energy_levels(n_states=4, n_frames=120, save_name='infinite_well_energy_levels')),
        ("ground state (n=1)", lambda: solver.animate_wavefunction_evolution(state_index=0, save_name='infinite_well_ground_state')),
        ("second state (n=2)", lambda: solver.animate_wavefunction_evolution(state_index=1, save_name='infinite_well_second_state')),
        ("superposition (n=1 + n=2)", lambda: solver.animate_superposition(state_indices=[0, 1], n_periods=3, n_frames=150, save_name='infinite_well_superposition')),
    ]

    for i, (name, func) in enumerate(animations):
        print(f"\n{i}. Animating {name}...")
        func()

    print("\n" + "=" * 60)
    print("Complete! Check 'gifs/' folder for all animations.")
    print("=" * 60)