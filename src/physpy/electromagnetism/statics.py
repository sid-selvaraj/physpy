"""
physpy.electromagnetism.statics
================================
Static electric field calculations and visualization.

Implements:
- PointCharge: isolated point charge electric field
- ElectricDipole: two equal and opposite charges
- ElectricField: superposition of multiple charges/dipoles
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from physpy.constants import epsilon_0


@dataclass
class PointCharge:
    """
    A point charge in 2D space.

    Parameters
    ----------
    q : float
        Charge in Coulombs (C).
    x : float, optional
        X position in meters. Default 0.0.
    y : float, optional
        Y position in meters. Default 0.0.
    """

    q: float
    x: float = 0.0
    y: float = 0.0

    def electric_field(self, x: float, y: float) -> tuple[float, float]:
        """
        Electric field at point (x, y) due to this charge.

        Returns
        -------
        Ex, Ey : electric field components, N/C
        """
        dx = x - self.x
        dy = y - self.y
        r2 = dx**2 + dy**2

        if r2 < 1e-12:
            return 0.0, 0.0

        r = np.sqrt(r2)
        k = 1 / (4 * np.pi * epsilon_0)
        magnitude = k * self.q / r2
        Ex = magnitude * dx / r
        Ey = magnitude * dy / r
        return Ex, Ey

    def potential(self, x: float, y: float) -> float:
        """Electric potential at point (x, y), in Volts."""
        dx = x - self.x
        dy = y - self.y
        r = np.sqrt(dx**2 + dy**2)

        if r < 1e-12:
            return float('inf') if self.q > 0 else float('-inf')

        k = 1 / (4 * np.pi * epsilon_0)
        return k * self.q / r


@dataclass
class ElectricDipole:
    """
    Two point charges of equal magnitude and opposite sign.

    Parameters
    ----------
    q : float
        Magnitude of each charge, Coulombs.
    x_pos : float
        X position of positive charge.
    y_pos : float
        Y position of positive charge.
    x_neg : float
        X position of negative charge.
    y_neg : float
        Y position of negative charge.
    """

    q: float
    x_pos: float
    y_pos: float
    x_neg: float
    y_neg: float

    def __post_init__(self):
        self._pos = PointCharge(q=self.q, x=self.x_pos, y=self.y_pos)
        self._neg = PointCharge(q=-self.q, x=self.x_neg, y=self.y_neg)

    @property
    def dipole_moment(self) -> float:
        """Dipole moment magnitude p = q * d."""
        d = np.sqrt((self.x_pos - self.x_neg)**2 + (self.y_pos - self.y_neg)**2)
        return self.q * d

    def electric_field(self, x: float, y: float) -> tuple[float, float]:
        """Electric field from superposition of both charges."""
        Ex_pos, Ey_pos = self._pos.electric_field(x, y)
        Ex_neg, Ey_neg = self._neg.electric_field(x, y)
        return Ex_pos + Ex_neg, Ey_pos + Ey_neg

    def potential(self, x: float, y: float) -> float:
        """Electric potential from superposition of both charges."""
        return self._pos.potential(x, y) + self._neg.potential(x, y)


@dataclass
class ElectricField:
    """
    Superposition of multiple point charges and dipoles.

    Build a composite field, evaluate it on a grid, and visualize.

    Parameters
    ----------
    charges : list of PointCharge, optional
        Point charges to include.
    dipoles : list of ElectricDipole, optional
        Dipoles to include.

    Examples
    --------
    >>> q1 = PointCharge(q=1e-6, x=-0.05, y=0)
    >>> q2 = PointCharge(q=-1e-6, x=0.05, y=0)
    >>> field = ElectricField(charges=[q1, q2])
    >>> Ex, Ey = field.electric_field(0, 0.1)
    """

    charges: list[PointCharge] = field(default_factory=list)
    dipoles: list[ElectricDipole] = field(default_factory=list)

    def electric_field(self, x: float, y: float) -> tuple[float, float]:
        """Total electric field at (x, y) from all charges and dipoles."""
        Ex, Ey = 0.0, 0.0
        for charge in self.charges:
            dEx, dEy = charge.electric_field(x, y)
            Ex += dEx
            Ey += dEy
        for dipole in self.dipoles:
            dEx, dEy = dipole.electric_field(x, y)
            Ex += dEx
            Ey += dEy
        return Ex, Ey

    def potential(self, x: float, y: float) -> float:
        """Total electric potential at (x, y)."""
        V = 0.0
        for charge in self.charges:
            V += charge.potential(x, y)
        for dipole in self.dipoles:
            V += dipole.potential(x, y)
        return V

    def field_grid(
        self,
        x_range: tuple[float, float] = (-0.1, 0.1),
        y_range: tuple[float, float] = (-0.1, 0.1),
        resolution: int = 30,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute electric field on a 2D grid.

        Returns
        -------
        X, Y, Ex, Ey : 2D arrays of grid points and field components
        """
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        Ex = np.zeros_like(X)
        Ey = np.zeros_like(Y)

        for i in range(resolution):
            for j in range(resolution):
                Ex[i, j], Ey[i, j] = self.electric_field(X[i, j], Y[i, j])

        return X, Y, Ex, Ey

    def potential_grid(
        self,
        x_range: tuple[float, float] = (-0.1, 0.1),
        y_range: tuple[float, float] = (-0.1, 0.1),
        resolution: int = 50,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute electric potential on a 2D grid.

        Returns
        -------
        X, Y, V : 2D arrays of grid points and potential
        """
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        V = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                V[i, j] = self.potential(X[i, j], Y[i, j])

        return X, Y, V

    def plot(self, ax=None, show: bool = True, field_arrows: bool = True, potential_contours: bool = True):
        """
        Visualize the electric field and equipotential lines.

        Parameters
        ----------
        ax : matplotlib axis, optional
            Axis to plot on. Creates new figure if None.
        show : bool
            Whether to call plt.show(). Default True.
        field_arrows : bool
            Whether to show field vector arrows. Default True.
        potential_contours : bool
            Whether to show equipotential contours. Default True.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 7))

        x_range = (-0.1, 0.1)
        y_range = (-0.1, 0.1)

        if potential_contours:
            X, Y, V = self.potential_grid(x_range, y_range, resolution=80)
            levels = np.linspace(-1e5, 1e5, 20)
            cs = ax.contour(X, Y, V, levels=levels, colors='gray', alpha=0.5, linewidths=0.5)

        if field_arrows:
            X, Y, Ex, Ey = self.field_grid(x_range, y_range, resolution=15)
            magnitude = np.sqrt(Ex**2 + Ey**2)
            magnitude[magnitude == 0] = 1
            ax.quiver(X, Y, Ex/magnitude, Ey/magnitude, magnitude, cmap='hot', scale=30)

        for charge in self.charges:
            color = 'red' if charge.q > 0 else 'blue'
            ax.plot(charge.x, charge.y, 'o', color=color, markersize=8, label=f'q={charge.q:.1e}')

        for dipole in self.dipoles:
            ax.plot(dipole.x_pos, dipole.y_pos, 'o', color='red', markersize=6)
            ax.plot(dipole.x_neg, dipole.y_neg, 'o', color='blue', markersize=6)

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title('Electric Field and Equipotential Lines')
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

        if show:
            plt.show()

        return ax
