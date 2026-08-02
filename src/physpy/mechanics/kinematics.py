"""
physpy.mechanics.kinematics
============================
Tools for kinematic simulation of moving bodies.

Currently implements 2D projectile motion, solved two ways:

- **Analytically**: the closed-form solution from introductory kinematics,
  valid only when air resistance is neglected.
- **Numerically**: by integrating the equations of motion with
  ``scipy.integrate.solve_ivp``. This supports quadratic air drag, and
  is structured so more forces (wind, the Magnus effect on a spinning
  ball, etc.) can be added later without changing the public interface.

Design note
-----------
There is one class, ``Projectile``, whose behavior (drag on or off) is
controlled by a parameter rather than split across two classes. Physically,
drag is just another term in F = ma, not a different kind of object — the
code should mirror that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from physpy.constants import g_earth


@dataclass
class Projectile:
    """
    A projectile launched in a uniform gravitational field, with optional
    quadratic air resistance.

    Parameters
    ----------
    v0 : float
        Launch speed, m/s.
    angle_deg : float
        Launch angle above the horizontal, degrees.
    mass : float, optional
        Mass of the projectile, kg. Only relevant when drag is enabled.
        Default 1.0 kg.
    drag_coefficient : float, optional
        Dimensionless drag coefficient C_d. Default 0.0 (no drag).
    cross_section_area : float, optional
        Cross-sectional area facing the direction of motion, m^2.
        Default 0.01 m^2 (roughly the frontal area of a baseball).
    air_density : float, optional
        Air density, kg/m^3. Default 1.225 (sea level, 15 C).
    gravity : float, optional
        Gravitational acceleration, m/s^2. Default Earth's g.

    Examples
    --------
    >>> p = Projectile(v0=20, angle_deg=45)
    >>> round(p.range_analytic(), 2)
    40.77
    """

    v0: float
    angle_deg: float
    mass: float = 1.0
    drag_coefficient: float = 0.0
    cross_section_area: float = 0.01
    air_density: float = 1.225
    gravity: float = g_earth

    _solution: dict | None = field(default=None, init=False, repr=False)

    @property
    def angle_rad(self) -> float:
        return np.radians(self.angle_deg)

    @property
    def vx0(self) -> float:
        return self.v0 * np.cos(self.angle_rad)

    @property
    def vy0(self) -> float:
        return self.v0 * np.sin(self.angle_rad)

    # ---------------------------------------------------------------
    # Analytic (no-drag) solution — this is our "ground truth" check
    # ---------------------------------------------------------------

    def time_of_flight_analytic(self) -> float:
        """Total flight time, assuming no air resistance, level ground."""
        return 2 * self.vy0 / self.gravity

    def range_analytic(self) -> float:
        """Horizontal range, assuming no air resistance, level ground."""
        return (self.v0 ** 2) * np.sin(2 * self.angle_rad) / self.gravity

    def max_height_analytic(self) -> float:
        """Maximum height, assuming no air resistance."""
        return (self.vy0 ** 2) / (2 * self.gravity)

    # ---------------------------------------------------------------
    # Numerical solution — supports drag
    # ---------------------------------------------------------------

    def _equations_of_motion(self, t, state):
        """
        state = [x, y, vx, vy]. Returns d(state)/dt.

        Quadratic drag opposes velocity:
            F_drag = -0.5 * rho * C_d * A * |v| * v
        """
        x, y, vx, vy = state
        speed = np.hypot(vx, vy)

        if self.drag_coefficient > 0 and speed > 0:
            drag_factor = (
                0.5 * self.air_density * self.drag_coefficient
                * self.cross_section_area * speed / self.mass
            )
            ax = -drag_factor * vx
            ay = -drag_factor * vy - self.gravity
        else:
            ax = 0.0
            ay = -self.gravity

        return [vx, vy, ax, ay]

    @staticmethod
    def _hit_ground(t, state):
        return state[1]  # y = 0

    def solve(self, t_max: float = 100.0, max_step: float = 0.01) -> dict:
        """
        Numerically integrate the trajectory from launch until the
        projectile returns to y = 0 (or t_max is reached).

        Returns
        -------
        dict with keys 't', 'x', 'y', 'vx', 'vy', each a numpy array.
        """
        state0 = [0.0, 0.0, self.vx0, self.vy0]

        hit_ground = self._hit_ground
        hit_ground.terminal = True
        hit_ground.direction = -1

        result = solve_ivp(
            self._equations_of_motion,
            t_span=(0, t_max),
            y0=state0,
            method="RK45",
            max_step=max_step,
            events=hit_ground,
            dense_output=True,
        )

        t, x, y, vx, vy = result.t, *result.y
        self._solution = {"t": t, "x": x, "y": y, "vx": vx, "vy": vy}
        return self._solution

    def plot(self, ax=None, compare_analytic: bool = True, show: bool = True):
        """
        Plot the trajectory. If drag is enabled and ``compare_analytic``
        is True, also overlay the no-drag analytic parabola — this makes
        the *cost* of air resistance visible at a glance.
        """
        import matplotlib.pyplot as plt

        if self._solution is None:
            self.solve()

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4.5))

        label = "Numerical (with drag)" if self.drag_coefficient > 0 else "Trajectory"
        ax.plot(self._solution["x"], self._solution["y"], label=label)

        if compare_analytic and self.drag_coefficient > 0:
            t_a = np.linspace(0, self.time_of_flight_analytic(), 200)
            x_a = self.vx0 * t_a
            y_a = self.vy0 * t_a - 0.5 * self.gravity * t_a ** 2
            ax.plot(x_a, y_a, "--", alpha=0.7, label="Analytic (no drag)")

        ax.set_xlabel("Horizontal distance (m)")
        ax.set_ylabel("Height (m)")
        ax.set_title(f"Projectile motion: v0={self.v0} m/s, angle={self.angle_deg}°")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

        if show:
            plt.show()

        return ax
