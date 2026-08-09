"""
physpy.quantum.schrodinger_1d
=============================
1D time-independent Schrödinger equation solver.

Solves -ℏ²/(2m) * d²ψ/dx² + V(x)ψ = E·ψ using the finite-difference method
and sparse eigenvalue solvers. Validates against analytic solutions for
particle-in-a-box and quantum harmonic oscillator.

Design note: Uses scipy.sparse.linalg.eigs for efficient eigenvalue computation
on large sparse matrices, enabling solution of realistic quantum systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal
from abc import ABC, abstractmethod

import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import eigs

from physpy.constants import hbar, m_electron


class Potential(ABC):
    """Base class for 1D quantum potential functions."""

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate potential at positions x."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Potential name for labeling."""
        pass


class ParticleInABox(Potential):
    """Infinite square well: V=0 inside [0, L], V=∞ elsewhere."""

    def __init__(self, L: float):
        self.L = L

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Potential is zero inside the box."""
        return np.zeros_like(x, dtype=float)

    def name(self) -> str:
        return f"Particle in a Box (L={self.L} m)"

    def energy_analytic(self, n: int) -> float:
        """Analytic energy levels: E_n = n²π²ℏ²/(2mL²)."""
        return (n**2 * np.pi**2 * hbar**2) / (2 * m_electron * self.L**2)


class QuantumHarmonicOscillator(Potential):
    """Harmonic oscillator: V(x) = 0.5 * m * ω² * x²."""

    def __init__(self, omega: float, mass: float = m_electron):
        self.omega = omega
        self.mass = mass

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Harmonic potential."""
        return 0.5 * self.mass * self.omega**2 * x**2

    def name(self) -> str:
        return f"Quantum Harmonic Oscillator (ω={self.omega:.2e} rad/s)"

    def energy_analytic(self, n: int) -> float:
        """Analytic energy levels: E_n = ℏ*ω*(n + 0.5)."""
        return hbar * self.omega * (n + 0.5)


class FiniteSquareWell(Potential):
    """Finite square well: V=0 inside [-L/2, L/2], V=V0 outside."""

    def __init__(self, L: float, V0: float):
        self.L = L
        self.V0 = V0

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Potential is V0 outside the well, 0 inside."""
        return np.where(np.abs(x) <= self.L / 2, 0.0, self.V0)

    def name(self) -> str:
        return f"Finite Square Well (L={self.L} m, V0={self.V0:.2e} J)"


class CustomPotential(Potential):
    """User-defined potential function."""

    def __init__(self, func: Callable[[np.ndarray], np.ndarray], name_str: str = "Custom"):
        self._func = func
        self._name = name_str

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate custom potential."""
        return self._func(x)

    def name(self) -> str:
        return self._name


@dataclass
class SchrodingerSolver1D:
    """
    Solves the 1D time-independent Schrödinger equation.

    Uses finite-difference method to convert the differential equation to
    a matrix eigenvalue problem, then uses sparse eigenvalue solver.

    Parameters
    ----------
    potential : Potential
        The potential energy function V(x).
    x_min, x_max : float
        Domain boundaries [x_min, x_max].
    num_points : int
        Number of grid points (resolution).
    mass : float, optional
        Particle mass in kg. Default: electron mass.
    num_states : int, optional
        Number of eigenvalues/eigenvectors to compute. Default 5.

    Examples
    --------
    >>> # Particle in a box
    >>> box = ParticleInABox(L=1e-9)
    >>> solver = SchrodingerSolver1D(box, x_min=0, x_max=1e-9)
    >>> energies, wavefunctions = solver.solve()
    >>> E1_numeric = energies[0]
    >>> E1_analytic = box.energy_analytic(1)
    """

    potential: Potential
    x_min: float
    x_max: float
    num_points: int = 500
    mass: float = m_electron
    num_states: int = 5

    def __post_init__(self):
        self.x = np.linspace(self.x_min, self.x_max, self.num_points)
        self.dx = self.x[1] - self.x[0]

        # Kinetic energy coefficient: -ℏ²/(2m*dx²)
        self._KE_coeff = -hbar**2 / (2 * self.mass * self.dx**2)

    def solve(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the Schrödinger equation.

        Returns
        -------
        energies : ndarray
            Sorted eigenvalues (energies) in Joules.
        wavefunctions : ndarray
            Eigenvectors (wavefunctions) with shape (num_points, num_states).
            Each column is a normalized wavefunction ψ_n(x).
        """
        # Evaluate potential on grid
        V_vals = self.potential.evaluate(self.x)

        # Build Hamiltonian matrix using sparse format
        # H = T + V where T is kinetic energy (tridiagonal), V is diagonal potential
        N = self.num_points

        # Tridiagonal part: -ℏ²/(2m*dx²) * d²/dx²
        # Central difference: d²ψ/dx² ≈ (ψ[i+1] - 2*ψ[i] + ψ[i-1]) / dx²
        diag_main = np.full(N, -2.0 * self._KE_coeff) + V_vals
        diag_upper = np.full(N - 1, self._KE_coeff)
        diag_lower = np.full(N - 1, self._KE_coeff)

        # Create sparse tridiagonal matrix
        H = diags(
            [diag_lower, diag_main, diag_upper],
            [-1, 0, 1],
            shape=(N, N),
            format="csr"
        )

        # Solve eigenvalue problem: find k smallest eigenvalues/eigenvectors
        k = min(self.num_states, N - 2)
        try:
            eigenvalues, eigenvectors = eigs(H, k=k, which='SM', return_eigenvectors=True)
        except Exception as e:
            raise ValueError(f"Eigenvalue solver failed: {e}")

        # Sort by energy (eigenvalues are real but may be unsorted)
        idx = np.argsort(eigenvalues.real)
        eigenvalues = eigenvalues[idx].real
        eigenvectors = eigenvectors[:, idx].real

        # Normalize wavefunctions: ∫|ψ|² dx = 1
        for i in range(eigenvectors.shape[1]):
            psi = eigenvectors[:, i]
            norm = np.sqrt(np.sum(np.abs(psi)**2) * self.dx)
            if norm > 0:
                eigenvectors[:, i] /= norm

        return eigenvalues, eigenvectors

    def energy_analytic(self, n: int) -> float:
        """
        Get analytic energy level if available.

        Only works for potentials with known solutions (ParticleInABox,
        QuantumHarmonicOscillator). Returns NaN for unknown potentials.
        """
        if isinstance(self.potential, ParticleInABox):
            return self.potential.energy_analytic(n)
        elif isinstance(self.potential, QuantumHarmonicOscillator):
            return self.potential.energy_analytic(n)
        else:
            return np.nan

    def probability_density(self, wavefunction: np.ndarray) -> np.ndarray:
        """
        Compute probability density ρ(x) = |ψ(x)|².

        Parameters
        ----------
        wavefunction : ndarray
            Wavefunction values at grid points.

        Returns
        -------
        ndarray
            Probability density ρ(x) = |ψ(x)|².
        """
        return np.abs(wavefunction)**2

    def plot(self, ax=None, show: bool = True, num_states: int = 3):
        """
        Plot the first num_states wavefunctions and potential.

        Parameters
        ----------
        ax : matplotlib axis, optional
            Axis to plot on. Creates new figure if None.
        show : bool
            Whether to call plt.show().
        num_states : int
            Number of wavefunctions to plot.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        # Solve to get wavefunctions
        energies, wavefunctions = self.solve()

        num_states = min(num_states, wavefunctions.shape[1])

        # Plot potential
        V_vals = self.potential.evaluate(self.x)
        ax.fill_between(self.x, 0, V_vals / np.max(V_vals[V_vals > 0] if np.any(V_vals > 0) else [1]),
                         alpha=0.2, color='gray', label='Potential V(x)')

        # Plot wavefunctions
        colors = plt.cm.viridis(np.linspace(0, 1, num_states))
        for i in range(num_states):
            psi = wavefunctions[:, i]
            E = energies[i]

            # Offset by energy level for visualization
            offset = E / np.max(energies) if np.max(energies) > 0 else 0
            ax.plot(self.x, psi + offset, color=colors[i], linewidth=2,
                   label=f'ψ₍{i+1}₎: E={E:.2e} J')

            # Mark energy level
            ax.axhline(y=offset, color=colors[i], linestyle='--', alpha=0.3, linewidth=1)

            # Optionally show analytic energy if available
            E_ana = self.energy_analytic(i + 1)
            if not np.isnan(E_ana):
                error_pct = abs(E - E_ana) / E_ana * 100
                ax.text(self.x[-1] * 0.98, offset, f' {error_pct:.1f}%',
                       fontsize=8, va='center', ha='right', color=colors[i])

        ax.set_xlabel('Position x (m)', fontsize=11, fontweight='bold')
        ax.set_ylabel('ψ(x) + Energy Offset (normalized)', fontsize=11, fontweight='bold')
        ax.set_title(f'1D Schrödinger Solver: {self.potential.name()}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(alpha=0.3)

        if show:
            plt.show()

        return ax
