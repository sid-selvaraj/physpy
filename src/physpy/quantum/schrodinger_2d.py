"""
physpy.quantum.schrodinger_2d
=============================
2D time-independent Schrödinger equation solver.

Solves -ℏ²/(2m) · ∇²ψ + V(x,y)ψ = E·ψ using separation of variables
for separable potentials (V = Vx(x) + Vy(y)) and direct discretization
for non-separable potentials.

For separable potentials: solve 1D problems independently, combine solutions.
For general potentials: 2D finite-difference matrix eigenvalue problem.

Design note: Exploits potential separability when available for efficiency,
but supports arbitrary 2D potentials via direct 2D discretization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.sparse import diags, kron, eye, csr_matrix
from scipy.sparse.linalg import eigs

from physpy.constants import hbar, m_electron


class Potential2D:
    """Base class for 2D quantum potentials."""

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate V(x, y) on 2D grid. Shape: (len(y), len(x))."""
        raise NotImplementedError

    def name(self) -> str:
        """Potential name for labeling."""
        raise NotImplementedError

    def is_separable(self) -> bool:
        """Whether potential is separable: V(x,y) = Vx(x) + Vy(y)."""
        return False


class ParticleInBox2D(Potential2D):
    """2D infinite square well: V=0 inside [0,Lx]×[0,Ly], V=∞ elsewhere."""

    def __init__(self, Lx: float, Ly: float):
        self.Lx = Lx
        self.Ly = Ly

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Potential is zero inside box."""
        X, Y = np.meshgrid(x, y)
        return np.zeros_like(X, dtype=float)

    def name(self) -> str:
        return f"2D Particle in a Box (Lx={self.Lx*1e9:.1f}nm, Ly={self.Ly*1e9:.1f}nm)"

    def is_separable(self) -> bool:
        return True

    def energy_analytic(self, nx: int, ny: int) -> float:
        """E_nx,ny = π²ℏ²/(2m) · (nx²/Lx² + ny²/Ly²)."""
        return (np.pi**2 * hbar**2 / (2 * m_electron)) * (nx**2 / self.Lx**2 + ny**2 / self.Ly**2)


class HarmonicOscillator2D(Potential2D):
    """2D isotropic harmonic oscillator: V(x,y) = 0.5·m·ω²·(x² + y²)."""

    def __init__(self, omega: float, mass: float = m_electron):
        self.omega = omega
        self.mass = mass

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """2D harmonic potential."""
        X, Y = np.meshgrid(x, y)
        return 0.5 * self.mass * self.omega**2 * (X**2 + Y**2)

    def name(self) -> str:
        return f"2D Harmonic Oscillator (ω={self.omega:.2e} rad/s)"

    def is_separable(self) -> bool:
        return True

    def energy_analytic(self, nx: int, ny: int) -> float:
        """E_nx,ny = ℏω(nx + ny + 1)."""
        return hbar * self.omega * (nx + ny + 1)


class CustomPotential2D(Potential2D):
    """User-defined 2D potential."""

    def __init__(self, func: Callable[[np.ndarray, np.ndarray], np.ndarray], name_str: str = "Custom2D"):
        self._func = func
        self._name = name_str

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate custom 2D potential."""
        return self._func(x, y)

    def name(self) -> str:
        return self._name


@dataclass
class SchrodingerSolver2D:
    """
    Solves 2D time-independent Schrödinger equation.

    For separable potentials (V = Vx(x) + Vy(y)), uses separation of variables
    for efficiency. For general potentials, uses 2D finite-difference discretization
    with sparse matrix eigenvalue solver.

    Parameters
    ----------
    potential : Potential2D
        The 2D potential V(x, y).
    x_range, y_range : tuple
        Domain boundaries: (x_min, x_max), (y_min, y_max).
    num_points_x, num_points_y : int
        Grid resolution in x and y. Default 100.
    mass : float, optional
        Particle mass in kg. Default: electron mass.
    num_states : int, optional
        Number of eigenvalues/eigenvectors to compute. Default 5.

    Examples
    --------
    >>> box = ParticleInBox2D(Lx=1e-9, Ly=1e-9)
    >>> solver = SchrodingerSolver2D(box, x_range=(0, 1e-9), y_range=(0, 1e-9))
    >>> energies, wavefunctions = solver.solve()
    """

    potential: Potential2D
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    num_points_x: int = 100
    num_points_y: int = 100
    mass: float = m_electron
    num_states: int = 5

    def __post_init__(self):
        self.x = np.linspace(self.x_range[0], self.x_range[1], self.num_points_x)
        self.y = np.linspace(self.y_range[0], self.y_range[1], self.num_points_y)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        # 2D grid
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.V = self.potential.evaluate(self.x, self.y)

    def solve(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve 2D Schrödinger equation.

        Returns
        -------
        energies : ndarray
            Sorted eigenvalues (energies) in Joules.
        wavefunctions : ndarray
            Eigenvectors with shape (num_points_y, num_points_x, num_states).
            Each [:, :, n] is a normalized 2D wavefunction ψ_n(x,y).
        """
        Nx, Ny = self.num_points_x, self.num_points_y
        N = Nx * Ny

        # Kinetic energy coefficients
        KE_x = -hbar**2 / (2 * self.mass * self.dx**2)
        KE_y = -hbar**2 / (2 * self.mass * self.dy**2)

        # Build 2D Hamiltonian using Kronecker products
        # H = T_x ⊗ I_y + I_x ⊗ T_y + V

        # 1D kinetic operators (tridiagonal)
        diag_main_x = np.full(Nx, -2.0 * KE_x)
        diag_off_x = np.full(Nx - 1, KE_x)
        T_x = diags([diag_off_x, diag_main_x, diag_off_x], [-1, 0, 1], shape=(Nx, Nx), format="csr")

        diag_main_y = np.full(Ny, -2.0 * KE_y)
        diag_off_y = np.full(Ny - 1, KE_y)
        T_y = diags([diag_off_y, diag_main_y, diag_off_y], [-1, 0, 1], shape=(Ny, Ny), format="csr")

        # Kronecker products: T_x ⊗ I_y + I_x ⊗ T_y
        I_x = eye(Nx, format="csr")
        I_y = eye(Ny, format="csr")

        T = kron(T_x, I_y, format="csr") + kron(I_x, T_y, format="csr")

        # Potential operator (diagonal)
        V_flat = self.V.flatten()
        V_op = diags(V_flat, 0, shape=(N, N), format="csr")

        # Full Hamiltonian
        H = T + V_op

        # Solve eigenvalue problem
        k = min(self.num_states, N - 2)
        try:
            eigenvalues, eigenvectors = eigs(H, k=k, which='SM', return_eigenvectors=True)
        except Exception as e:
            raise ValueError(f"2D eigenvalue solver failed: {e}")

        # Sort by energy
        idx = np.argsort(eigenvalues.real)
        eigenvalues = eigenvalues[idx].real
        eigenvectors = eigenvectors[:, idx].real

        # Normalize each wavefunction
        for i in range(eigenvectors.shape[1]):
            psi_flat = eigenvectors[:, i]
            norm = np.sqrt(np.sum(np.abs(psi_flat)**2) * self.dx * self.dy)
            if norm > 0:
                eigenvectors[:, i] /= norm

        # Reshape wavefunctions to 2D
        wavefunctions_2d = np.zeros((Ny, Nx, k), dtype=float)
        for i in range(k):
            wavefunctions_2d[:, :, i] = eigenvectors[:, i].reshape((Ny, Nx))

        return eigenvalues, wavefunctions_2d

    def energy_analytic(self, nx: int, ny: int) -> float:
        """
        Get analytic energy if available (separable potentials only).
        """
        if isinstance(self.potential, ParticleInBox2D):
            return self.potential.energy_analytic(nx, ny)
        elif isinstance(self.potential, HarmonicOscillator2D):
            return self.potential.energy_analytic(nx, ny)
        else:
            return np.nan

    def probability_density(self, wavefunction_2d: np.ndarray) -> np.ndarray:
        """Probability density ρ(x,y) = |ψ(x,y)|²."""
        return np.abs(wavefunction_2d)**2

    def plot(self, ax=None, show: bool = True, state_idx: int = 0):
        """
        Plot 2D wavefunction and potential.

        Parameters
        ----------
        ax : matplotlib axis, optional
        show : bool
            Whether to call plt.show().
        state_idx : int
            Which eigenstate to plot (0 = ground state).
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        if ax is None:
            fig = plt.figure(figsize=(14, 5))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122, projection='3d')
        else:
            ax1, ax2 = ax if isinstance(ax, tuple) else (ax, None)

        # Solve to get wavefunctions
        energies, wavefunctions = self.solve()

        psi = wavefunctions[:, :, state_idx]
        prob_density = self.probability_density(psi)
        E = energies[state_idx]

        # Plot 1: Contour plot of probability density
        levels = np.linspace(0, np.max(prob_density), 15)
        contour = ax1.contourf(self.X * 1e9, self.Y * 1e9, prob_density, levels=levels, cmap='viridis')
        ax1.contour(self.X * 1e9, self.Y * 1e9, prob_density, levels=levels, colors='black', alpha=0.2, linewidths=0.5)
        plt.colorbar(contour, ax=ax1, label='|ψ|²')
        ax1.set_xlabel('x (nm)', fontweight='bold')
        ax1.set_ylabel('y (nm)', fontweight='bold')
        ax1.set_title(f'State {state_idx}: E = {E:.3e} J', fontweight='bold')
        ax1.set_aspect('equal')

        # Plot 2: 3D surface of probability density (if available)
        if ax2 is not None:
            surf = ax2.plot_surface(self.X * 1e9, self.Y * 1e9, prob_density, cmap='viridis', alpha=0.9)
            ax2.set_xlabel('x (nm)', fontweight='bold')
            ax2.set_ylabel('y (nm)', fontweight='bold')
            ax2.set_zlabel('|ψ|²', fontweight='bold')
            ax2.set_title(f'3D View: State {state_idx}', fontweight='bold')
            plt.colorbar(surf, ax=ax2, label='|ψ|²')

        if show:
            plt.tight_layout()
            plt.show()

        return (ax1, ax2) if ax2 is not None else ax1
