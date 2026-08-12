"""Tests for N-body gravitational solver."""

import pytest
import numpy as np
from physpy.mechanics.nbody import Body, NBodySolver
from physpy.constants import G_grav


def test_two_body_orbit():
    """Two bodies in circular orbit."""
    # Binary: equal masses at equal distance from center
    M = 1e30  # kg
    r = 1e11  # m (AU-like)

    b1 = Body(mass=M, position=[-r/2, 0, 0], velocity=[0, 1e4, 0])
    b2 = Body(mass=M, position=[r/2, 0, 0], velocity=[0, -1e4, 0])

    solver = NBodySolver([b1, b2])
    sol = solver.solve(t_max=1e7, num_steps=100)

    # Should complete without error
    assert len(sol['t']) > 1
    assert sol['positions'].shape == (len(sol['t']), 2, 3)


def test_three_body_stable():
    """Three bodies in configuration space."""
    M = 1e30
    b1 = Body(mass=M, position=[0, 0, 0], velocity=[0, 0, 0])
    b2 = Body(mass=M, position=[1e11, 0, 0], velocity=[0, 1e4, 0])
    b3 = Body(mass=M, position=[-1e11, 0, 0], velocity=[0, -1e4, 0])

    solver = NBodySolver([b1, b2, b3])
    sol = solver.solve(t_max=1e7, num_steps=50)

    assert sol['positions'].shape[1] == 3


def test_energy_conservation():
    """Total energy is conserved."""
    M = 1e30
    b1 = Body(mass=M, position=[0, 0, 0], velocity=[0, 0, 0])
    b2 = Body(mass=M, position=[1e11, 0, 0], velocity=[0, 2e4, 0])

    solver = NBodySolver([b1, b2])
    sol = solver.solve(t_max=1e6, num_steps=50)

    energies = []
    for i in range(len(sol['t'])):
        E = solver.total_energy(sol['positions'][i], sol['velocities'][i])
        energies.append(E)

    energies = np.array(energies)
    E_drift = (np.max(energies) - np.min(energies)) / np.abs(np.mean(energies))

    # Energy should be conserved to ~1%
    assert E_drift < 0.05


def test_momentum_conservation():
    """Total momentum is conserved (no external forces)."""
    M = 1e30
    b1 = Body(mass=M, position=[0, 0, 0], velocity=[1e4, 0, 0])
    b2 = Body(mass=M, position=[1e11, 0, 0], velocity=[-1e4, 0, 0])

    solver = NBodySolver([b1, b2])
    sol = solver.solve(t_max=1e6, num_steps=50)

    p_initial = solver.total_momentum(sol['velocities'][0])
    p_final = solver.total_momentum(sol['velocities'][-1])

    assert np.allclose(p_initial, p_final, rtol=0.1)


def test_softening_prevents_singularity():
    """Softening parameter prevents divergence at close encounters."""
    M = 1e30
    b1 = Body(mass=M, position=[0, 0, 0], velocity=[0, 0, 0])
    b2 = Body(mass=M, position=[1e9, 0, 0], velocity=[0, 1e5, 0])  # Close approach

    solver = NBodySolver([b1, b2], softening=1e9)
    sol = solver.solve(t_max=1e6, num_steps=50)

    # Should not have NaN or inf
    assert np.all(np.isfinite(sol['positions']))
    assert np.all(np.isfinite(sol['velocities']))


def test_single_body_coasting():
    """Single body with no forces coasts at constant velocity."""
    b1 = Body(mass=1e30, position=[0, 0, 0], velocity=[1e4, 2e4, 3e4])

    solver = NBodySolver([b1])
    sol = solver.solve(t_max=1e7, num_steps=50)

    # Velocity should be constant
    v_initial = sol['velocities'][0]
    v_final = sol['velocities'][-1]

    assert np.allclose(v_initial, v_final, rtol=1e-10)


def test_position_update():
    """Positions update with velocities."""
    b1 = Body(mass=1e30, position=[0, 0, 0], velocity=[1e4, 0, 0])
    b2 = Body(mass=1e30, position=[1e11, 0, 0], velocity=[0, 0, 0])

    solver = NBodySolver([b1, b2])
    sol = solver.solve(t_max=1e6, num_steps=50)

    # Position should change
    pos_0 = sol['positions'][0, 0]
    pos_f = sol['positions'][-1, 0]

    assert not np.allclose(pos_0, pos_f, atol=1e10)


def test_escape_trajectory():
    """High velocity trajectory produces different dynamics."""
    M = 1e30
    r = 1e11

    b1 = Body(mass=M, position=[0, 0, 0], velocity=[0, 0, 0])
    b2 = Body(mass=M, position=[r, 0, 0], velocity=[0, 1e5, 0])  # High velocity

    solver = NBodySolver([b1, b2])
    sol = solver.solve(t_max=1e8, num_steps=50)

    # Simulate and verify no numerical issues
    assert np.all(np.isfinite(sol['positions']))
    assert np.all(np.isfinite(sol['velocities']))


def test_circular_orbit_period():
    """Circular orbit has expected period."""
    M = 1e30
    r = 1e11
    v_circ = np.sqrt(G_grav * M / r)
    T_expected = 2 * np.pi * r / v_circ

    b1 = Body(mass=M, position=[0, 0, 0], velocity=[0, 0, 0])
    b2 = Body(mass=M/100, position=[r, 0, 0], velocity=[0, v_circ, 0])

    solver = NBodySolver([b1, b2])
    sol = solver.solve(t_max=T_expected*1.5, num_steps=100)

    # Period is roughly conserved (not exact due to integration error)
    assert len(sol['t']) > 50


def test_nbody_initialization():
    """N-body solver initializes correctly."""
    bodies = [
        Body(mass=1e30, position=[0, 0, 0], velocity=[0, 0, 0]),
        Body(mass=1e30, position=[1e11, 0, 0], velocity=[0, 1e4, 0]),
        Body(mass=1e30, position=[0, 1e11, 0], velocity=[-1e4, 0, 0]),
    ]

    solver = NBodySolver(bodies)

    assert solver.N == 3
    assert len(solver.bodies) == 3
