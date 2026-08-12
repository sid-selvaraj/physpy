"""Tests for 1D EM wave FDTD solver."""

import pytest
import numpy as np
from physpy.electromagnetism.waves_em import FDTDSolver1D
from physpy.constants import c_light, epsilon_0, mu_0


def test_fdtd_initialization():
    """FDTD solver initializes with correct grid."""
    solver = FDTDSolver1D(x_min=0, x_max=1.0, num_points=256)

    assert len(solver.x) == 256
    assert solver.x[0] == 0
    assert np.isclose(solver.x[-1], 1.0)
    assert len(solver.E) == 256
    assert len(solver.B) == 256


def test_fdtd_timestep_valid():
    """Timestep satisfies Courant condition."""
    solver = FDTDSolver1D(num_points=256, cfl=0.99)

    # dt < dx / c (Courant condition)
    assert solver.dt < solver.dx / c_light


def test_wave_speed_property():
    """Wave speed equals c."""
    solver = FDTDSolver1D(x_min=0, x_max=1.0, num_points=256)

    assert np.isclose(solver.wave_speed(), c_light)


def test_energy_density_calculation():
    """Energy density is computed correctly."""
    solver = FDTDSolver1D(x_min=0, x_max=1.0, num_points=128)

    E_test = np.ones(128)
    B_test = np.ones(128)

    u = solver.energy_density(E_test, B_test)

    expected = 0.5 * (epsilon_0 * 1.0**2 + 1.0**2 / mu_0)
    assert np.allclose(u, expected)


def test_poynting_vector_calculation():
    """Poynting vector is computed correctly."""
    solver = FDTDSolver1D(x_min=0, x_max=1.0, num_points=128)

    E_test = np.ones(128) * 2.0
    B_test = np.ones(128) * 0.5

    S = solver.poynting_vector(E_test, B_test)

    expected = 2.0 * 0.5 / mu_0
    assert np.allclose(S, expected)


def test_fdtd_short_simulation():
    """Short FDTD simulation completes without error."""
    solver = FDTDSolver1D(x_min=0, x_max=0.5, num_points=64)

    sol = solver.solve(t_max=0.01)  # Very short simulation

    assert len(sol['t']) > 1
    assert np.all(np.isfinite(sol['E']))
    assert np.all(np.isfinite(sol['B']))


def test_source_function_callback():
    """Custom source function works."""
    solver = FDTDSolver1D(x_min=0, x_max=1.0, num_points=64)

    def custom_source(t):
        return np.sin(2 * np.pi * 100 * t)

    sol = solver.solve(t_max=0.01, source_func=custom_source)

    # Should have results
    assert len(sol['t']) > 1


def test_field_array_shapes():
    """Output arrays have correct shapes."""
    solver = FDTDSolver1D(x_min=0, x_max=1.0, num_points=128)

    sol = solver.solve(t_max=0.01)

    assert sol['E'].shape[1] == 128
    assert sol['B'].shape[1] == 128
    assert len(sol['E']) == len(sol['t'])
    assert len(sol['B']) == len(sol['t'])


def test_maxwell_equation_consistency():
    """E and B evolve together."""
    solver = FDTDSolver1D(x_min=0, x_max=1.0, num_points=64)

    sol = solver.solve(t_max=0.02)

    # E and B should both be generated
    assert not np.allclose(sol['E'], 0)
    assert not np.allclose(sol['B'], 0)


def test_different_grid_resolutions():
    """Solver works with different grid sizes."""
    for num_points in [32, 64, 128, 256]:
        solver = FDTDSolver1D(x_min=0, x_max=1.0, num_points=num_points)
        sol = solver.solve(t_max=0.005)

        assert sol['E'].shape == (len(sol['t']), num_points)
