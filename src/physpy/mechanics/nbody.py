"""
physpy.mechanics.nbody
======================
N-body gravitational simulation using Runge-Kutta integration.

Solves coupled ODEs for gravitational interactions between N point masses.
Validates energy and momentum conservation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.integrate import solve_ivp

from physpy.constants import G_grav


@dataclass
class Body:
    """A massive point particle."""
    mass: float  # kg
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))

    def __post_init__(self):
        self.position = np.array(self.position, dtype=float)
        self.velocity = np.array(self.velocity, dtype=float)


class NBodySolver:
    """Solves N-body gravitational dynamics."""

    def __init__(self, bodies: list[Body], softening: float = 0.0):
        """
        Initialize N-body solver.

        Parameters
        ----------
        bodies : list[Body]
            List of massive bodies.
        softening : float
            Softening parameter to prevent singularities. Default 0 (exact).
        """
        self.bodies = bodies
        self.softening = softening
        self.N = len(bodies)

    def _equations_of_motion(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Compute accelerations for N bodies.

        state = [x1, y1, z1, vx1, vy1, vz1, ..., xN, yN, zN, vxN, vyN, vzN]
        Returns dstate/dt
        """
        # Unpack positions and velocities
        positions = state[:3*self.N].reshape((self.N, 3))
        velocities = state[3*self.N:].reshape((self.N, 3))

        # Compute accelerations
        accelerations = np.zeros((self.N, 3))

        for i in range(self.N):
            for j in range(i+1, self.N):
                r_ij = positions[j] - positions[i]
                r_mag = np.linalg.norm(r_ij)
                r_mag_soft = np.sqrt(r_mag**2 + self.softening**2)

                # Gravitational force on i from j
                F_mag = G_grav * self.bodies[i].mass * self.bodies[j].mass / r_mag_soft**3

                accelerations[i] += F_mag * r_ij / self.bodies[i].mass
                accelerations[j] -= F_mag * r_ij / self.bodies[j].mass

        # Return state derivative: [velocities; accelerations]
        return np.concatenate([velocities.flatten(), accelerations.flatten()])

    def solve(self, t_max: float, num_steps: int = 100) -> dict:
        """
        Integrate N-body system to time t_max.

        Returns
        -------
        dict with 't', 'positions', 'velocities' (all as arrays over time)
        """
        # Initial state
        state0 = np.concatenate([
            np.concatenate([b.position for b in self.bodies]),
            np.concatenate([b.velocity for b in self.bodies])
        ])

        # Integrate
        result = solve_ivp(
            self._equations_of_motion,
            (0, t_max),
            state0,
            method='RK45',
            dense_output=True,
            max_step=t_max/num_steps
        )

        # Extract positions and velocities
        t = result.t
        positions = np.zeros((len(t), self.N, 3))
        velocities = np.zeros((len(t), self.N, 3))

        for i, t_i in enumerate(t):
            s = result.sol(t_i)
            positions[i] = s[:3*self.N].reshape((self.N, 3))
            velocities[i] = s[3*self.N:].reshape((self.N, 3))

        return {'t': t, 'positions': positions, 'velocities': velocities}

    def total_energy(self, positions: np.ndarray, velocities: np.ndarray) -> float:
        """
        Compute total energy: E = KE + PE.

        Parameters
        ----------
        positions : ndarray, shape (N, 3)
        velocities : ndarray, shape (N, 3)
        """
        # Kinetic energy
        KE = 0.5 * np.sum([self.bodies[i].mass * np.linalg.norm(velocities[i])**2
                            for i in range(self.N)])

        # Potential energy
        PE = 0.0
        for i in range(self.N):
            for j in range(i+1, self.N):
                r_ij = np.linalg.norm(positions[j] - positions[i])
                if r_ij > 1e-10:
                    PE -= G_grav * self.bodies[i].mass * self.bodies[j].mass / r_ij

        return KE + PE

    def total_momentum(self, velocities: np.ndarray) -> np.ndarray:
        """Total linear momentum: p = Σ m_i * v_i."""
        return np.sum([self.bodies[i].mass * velocities[i] for i in range(self.N)], axis=0)
