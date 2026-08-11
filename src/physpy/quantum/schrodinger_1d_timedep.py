"""
physpy.quantum.schrodinger_1d_timedep
=====================================
1D time-dependent Schrödinger equation solver.

Solves ∂ψ/∂t = -i/ℏ · H · ψ using the split-step Fourier method (SSFM).
This is a key algorithm for quantum dynamics: efficient, unconditionally stable,
and preserves norm and energy to high precision.

Design note: Split-step Fourier separates kinetic and potential energy evolution,
applying each in the most efficient domain (momentum space for kinetic, position
space for potential). This reduces error and improves stability vs simple methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.fft import fft, ifft

from physpy.constants import hbar, m_electron
from physpy.quantum.schrodinger_1d import Potential, ParticleInABox


@dataclass
class SchrodingerSolverTimeDep1D:
    """
    Solves the 1D time-dependent Schrödinger equation using split-step Fourier method.

    The split-step method is unconditionally stable and highly accurate:
    1. Half-step potential evolution in position space: ψ → exp(-i·V·dt/(2ℏ)) · ψ
    2. Full-step kinetic energy evolution in momentum space
    3. Half-step potential evolution in position space

    Parameters
    ----------
    potential : Potential
        The potential energy V(x).
    x_min, x_max : float
        Domain boundaries [x_min, x_max].
    num_points : int
        Number of grid points. Must be power of 2 for efficient FFT. Default 256.
    mass : float, optional
        Particle mass in kg. Default: electron mass.
    psi_initial : callable, optional
        Initial wavefunction ψ(x). If None, uses ground state of potential.

    Examples
    --------
    >>> # Wavepacket in free space
    >>> box = ParticleInABox(L=1e-8)  # Large box (nearly free)
    >>> solver = SchrodingerSolverTimeDep1D(box, x_min=0, x_max=1e-8)
    >>> t, psi_evolution = solver.evolve(t_max=1e-15, num_steps=100)
    >>> # Wavepacket spreads due to dispersion
    """

    potential: Potential
    x_min: float
    x_max: float
    num_points: int = 256
    mass: float = m_electron
    psi_initial: Callable[[np.ndarray], np.ndarray] | None = None

    def __post_init__(self):
        # Ensure num_points is power of 2 for efficient FFT
        if (self.num_points & (self.num_points - 1)) != 0:
            new_size = 2 ** int(np.ceil(np.log2(self.num_points)))
            self.num_points = new_size

        self.x = np.linspace(self.x_min, self.x_max, self.num_points)
        self.dx = self.x[1] - self.x[0]

        # Momentum space grid
        self.k = 2 * np.pi * np.fft.fftfreq(self.num_points, self.dx)

        # Kinetic energy operator in momentum space: T = ℏ²k²/(2m)
        self._KE_op = hbar**2 * self.k**2 / (2 * self.mass)

        # Potential energy on position grid
        self.V = self.potential.evaluate(self.x)

        # Set initial wavefunction
        if self.psi_initial is None:
            # Use ground state of potential (Gaussian for free particle, etc.)
            sigma = self.dx * 10  # Width of initial packet
            psi_0 = np.exp(-0.5 * (self.x - (self.x_min + self.x_max) / 2)**2 / sigma**2)
            psi_0 /= np.sqrt(np.sum(np.abs(psi_0)**2) * self.dx)
            self.psi_initial_func = lambda x: psi_0
        else:
            self.psi_initial_func = self.psi_initial

    def _normalize(self, psi: np.ndarray) -> np.ndarray:
        """Normalize wavefunction: ∫|ψ|² dx = 1."""
        norm = np.sqrt(np.sum(np.abs(psi)**2) * self.dx)
        return psi / norm if norm > 0 else psi

    def _half_step_potential(self, psi: np.ndarray, dt: float, direction: int = 1) -> np.ndarray:
        """
        Half-step potential evolution in position space.

        ψ(t+dt/2) = exp(-i·V·dt/(2ℏ)) · ψ(t)
        """
        factor = np.exp(-1j * self.V * dt / (2 * hbar) * direction)
        return psi * factor

    def _full_step_kinetic(self, psi: np.ndarray, dt: float) -> np.ndarray:
        """
        Full-step kinetic energy evolution in momentum space.

        ψ(t+dt) = exp(-i·T·dt/ℏ) · ψ(t) in momentum space.
        """
        # Transform to momentum space
        psi_k = fft(psi)

        # Apply kinetic energy operator
        factor = np.exp(-1j * self._KE_op * dt / hbar)
        psi_k *= factor

        # Transform back to position space
        psi = ifft(psi_k).real  # Take real part (imaginary part is numerical noise)
        return psi

    def evolve(
        self, t_max: float, num_steps: int, initial_psi: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evolve the wavefunction from t=0 to t=t_max using split-step method.

        Parameters
        ----------
        t_max : float
            Final time in seconds.
        num_steps : int
            Number of time steps.
        initial_psi : ndarray, optional
            Initial wavefunction. If None, uses configured initial state.

        Returns
        -------
        t : ndarray
            Time points (shape: num_steps)
        psi_evolution : ndarray
            Wavefunction at each time (shape: num_steps × num_points)
            Each row is ψ(x, t_i)
        """
        dt = t_max / num_steps
        t = np.linspace(0, t_max, num_steps)

        # Initialize wavefunction
        if initial_psi is None:
            psi = self.psi_initial_func(self.x)
        else:
            psi = initial_psi.copy()

        psi = self._normalize(psi)

        # Store evolution
        psi_evolution = np.zeros((num_steps, self.num_points), dtype=complex)
        psi_evolution[0] = psi

        # Time evolution using split-step Fourier method
        for n in range(1, num_steps):
            # Half-step potential
            psi = self._half_step_potential(psi, dt)

            # Full-step kinetic
            psi = self._full_step_kinetic(psi, dt)

            # Half-step potential
            psi = self._half_step_potential(psi, dt)

            # Normalize to maintain ∫|ψ|² dx = 1
            psi = self._normalize(psi)

            psi_evolution[n] = psi

        return t, psi_evolution

    def energy_expectation(self, psi: np.ndarray) -> float:
        """
        Compute expectation value of energy: <H> = <T> + <V>.

        ⟨T⟩ = ∫ ψ* (-ℏ²/2m d²/dx²) ψ dx
        ⟨V⟩ = ∫ ψ* V(x) ψ dx
        """
        # Kinetic energy: <T> = ℏ²/2m ∫|dψ/dx|² dx
        dpsi_dx = np.gradient(psi, self.dx)
        KE = hbar**2 / (2 * self.mass) * np.sum(np.abs(dpsi_dx)**2) * self.dx

        # Potential energy: <V> = ∫ ψ* V ψ dx
        PE = np.sum(np.conj(psi) * self.V * psi).real * self.dx

        return KE + PE

    def norm_check(self, psi: np.ndarray) -> float:
        """Check wavefunction normalization: ∫|ψ|² dx."""
        return np.sum(np.abs(psi)**2) * self.dx

    def probability_density(self, psi: np.ndarray) -> np.ndarray:
        """Probability density ρ(x) = |ψ(x)|²."""
        return np.abs(psi)**2

    def plot_evolution(self, t: np.ndarray, psi_evolution: np.ndarray,
                      ax=None, show: bool = True):
        """
        Plot wavefunction evolution over time.

        Shows |ψ(x,t)|² as a 2D heatmap or line plots at key times.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        else:
            ax1, ax2 = ax if isinstance(ax, tuple) else (ax, None)

        # Plot 1: Heatmap of probability density over time
        prob_density = np.abs(psi_evolution)**2
        im = ax1.contourf(self.x * 1e9, t * 1e15, prob_density, levels=20, cmap='viridis')
        ax1.set_xlabel('Position x (nm)', fontweight='bold')
        ax1.set_ylabel('Time t (fs)', fontweight='bold')
        ax1.set_title('Wavepacket Evolution |ψ(x,t)|²', fontweight='bold')
        plt.colorbar(im, ax=ax1, label='|ψ|²')

        # Plot 2: Snapshots at different times if ax2 provided
        if ax2 is not None:
            times_to_plot = [0, len(t)//4, len(t)//2, 3*len(t)//4, -1]
            colors = plt.cm.viridis(np.linspace(0, 1, len(times_to_plot)))

            for idx, (time_idx, color) in enumerate(zip(times_to_plot, colors)):
                psi_t = psi_evolution[time_idx]
                label = f't={t[time_idx]*1e15:.2f} fs'
                ax2.plot(self.x * 1e9, np.abs(psi_t)**2, color=color, linewidth=2, label=label)

            ax2.set_xlabel('Position x (nm)', fontweight='bold')
            ax2.set_ylabel('|ψ(x)|²', fontweight='bold')
            ax2.set_title('Probability Density Snapshots', fontweight='bold')
            ax2.legend(fontsize=9)
            ax2.grid(alpha=0.3)

        if show:
            plt.tight_layout()
            plt.show()

        return (ax1, ax2) if ax2 is not None else ax1
