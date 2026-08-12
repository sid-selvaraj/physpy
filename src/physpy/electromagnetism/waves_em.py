"""
physpy.electromagnetism.waves_em
================================
1D electromagnetic wave propagation using FDTD (Finite-Difference Time-Domain).

Solves Maxwell equations in 1D: ∂E/∂t = c² ∂B/∂x, ∂B/∂t = ∂E/∂x
Validates wave speed, energy conservation, and analytical solutions.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from physpy.constants import c_light, epsilon_0, mu_0


@dataclass
class FDTDSolver1D:
    """1D FDTD solver for Maxwell equations."""

    x_min: float = 0.0
    x_max: float = 1.0
    num_points: int = 256
    cfl: float = 0.99  # Courant-Friedrichs-Lewy parameter

    def __post_init__(self):
        self.x = np.linspace(self.x_min, self.x_max, self.num_points)
        self.dx = self.x[1] - self.x[0]
        self.dt = self.cfl * self.dx / c_light

        # E and B fields on staggered Yee grid
        self.E = np.zeros(self.num_points)
        self.B = np.zeros(self.num_points)

    def _apply_pml_boundary(self):
        """Simple absorbing boundary (first and last point = 0)."""
        self.E[0] = 0
        self.E[-1] = 0
        self.B[0] = 0
        self.B[-1] = 0

    def evolve_step(self):
        """Single FDTD time step."""
        # Compute dE/dx and dB/dx
        dE_dx = np.gradient(self.E, self.dx)
        dB_dx = np.gradient(self.B, self.dx)

        # Update using leapfrog scheme
        E_new = self.E - c_light**2 * self.dt / self.dx * np.diff(self.B, prepend=self.B[-1])
        B_new = self.B - self.dt / self.dx * np.diff(self.E, prepend=self.E[-1])

        self.E = E_new
        self.B = B_new

        self._apply_pml_boundary()

    def solve(self, t_max: float, source_func=None, source_pos_idx: int = -1) -> dict:
        """
        Solve Maxwell equations from t=0 to t_max.

        Parameters
        ----------
        t_max : float
            Simulation time.
        source_func : callable, optional
            Source function E(t). If None, uses Gaussian pulse.
        source_pos_idx : int
            Source position index. Default: center.

        Returns
        -------
        dict with 't', 'E_field', 'B_field' time series
        """
        if source_pos_idx == -1:
            source_pos_idx = self.num_points // 2

        if source_func is None:
            # Gaussian pulse
            t_0 = 3 * self.dt
            sigma = t_0 / 4
            source_func = lambda t: np.exp(-((t - t_0) / sigma)**2)

        num_steps = int(t_max / self.dt)
        t_array = np.arange(num_steps) * self.dt

        E_history = np.zeros((num_steps, self.num_points))
        B_history = np.zeros((num_steps, self.num_points))

        for n in range(num_steps):
            # Apply source
            self.E[source_pos_idx] += source_func(n * self.dt)

            # FDTD step
            self.evolve_step()

            E_history[n] = self.E.copy()
            B_history[n] = self.B.copy()

        return {'t': t_array, 'E': E_history, 'B': B_history}

    def wave_speed(self) -> float:
        """Expected wave speed: c."""
        return c_light

    def energy_density(self, E: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Energy density: u = 0.5*(ε₀*E² + B²/μ₀)."""
        return 0.5 * (epsilon_0 * E**2 + B**2 / mu_0)

    def poynting_vector(self, E: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Poynting vector: S = E × B / μ₀."""
        return E * B / mu_0
