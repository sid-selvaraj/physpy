"""Tests for physpy.quantum.schrodinger_2d.

Validates 2D Schrödinger solver against known analytic solutions
for 2D particle-in-a-box and 2D harmonic oscillator.
"""

import pytest
import numpy as np

from physpy.quantum.schrodinger_2d import (
    SchrodingerSolver2D, ParticleInBox2D, HarmonicOscillator2D, CustomPotential2D
)
from physpy.constants import hbar, m_electron


class TestParticleInBox2D:
    """Test 2D particle-in-a-box solver."""

    def test_box_ground_state_energy(self):
        """Ground state (1,1) matches analytic: E = π²ℏ²(1/Lx² + 1/Ly²)/(2m)."""
        Lx, Ly = 1e-9, 1e-9  # 1 nm square box
        box = ParticleInBox2D(Lx=Lx, Ly=Ly)
        solver = SchrodingerSolver2D(
            box,
            x_range=(0, Lx),
            y_range=(0, Ly),
            num_points_x=100,
            num_points_y=100,
            num_states=3
        )

        energies, _ = solver.solve()
        E_num = energies[0]
        E_ana = box.energy_analytic(1, 1)

        # Finite-difference error: ~1% is typical
        assert np.isclose(E_num, E_ana, rtol=0.02)

    def test_box_first_excited_states(self):
        """Test (2,1) and (1,2) states are degenerate for square box."""
        Lx = Ly = 1e-9
        box = ParticleInBox2D(Lx=Lx, Ly=Ly)
        solver = SchrodingerSolver2D(
            box,
            x_range=(0, Lx),
            y_range=(0, Ly),
            num_points_x=100,
            num_points_y=100,
            num_states=5
        )

        energies, _ = solver.solve()

        # For square box: E_1,1, then degenerate E_2,1 = E_1,2
        # Find first excited level (should be double degenerate)
        E_ground = energies[0]

        # States 1 and 2 should be close (both ~ E_2,1)
        E_ana_21 = box.energy_analytic(2, 1)

        # Check that we find this energy level
        assert any(np.isclose(E, E_ana_21, rtol=0.02) for E in energies[:3])

    def test_box_rectangular_anisotropy(self):
        """Rectangular box: E ∝ (nx²/Lx² + ny²/Ly²)."""
        Lx, Ly = 1e-9, 2e-9  # 1:2 aspect ratio
        box = ParticleInBox2D(Lx=Lx, Ly=Ly)
        solver = SchrodingerSolver2D(
            box,
            x_range=(0, Lx),
            y_range=(0, Ly),
            num_points_x=100,
            num_points_y=100,
            num_states=5
        )

        energies, _ = solver.solve()

        # For Ly = 2*Lx: E ∝ (nx²/Lx² + ny²/(4Lx²))
        # E_1,1 ∝ (1 + 0.25) = 1.25
        # E_1,2 ∝ (1 + 1) = 2.0
        # E_2,1 ∝ (4 + 0.25) = 4.25
        # Order: E_1,1 < E_1,2 < E_2,1
        E_11 = box.energy_analytic(1, 1)
        E_12 = box.energy_analytic(1, 2)
        E_21 = box.energy_analytic(2, 1)

        assert E_11 < E_12 < E_21

        # Verify solver finds them in correct order
        assert np.isclose(energies[0], E_11, rtol=0.02)
        assert np.isclose(energies[1], E_12, rtol=0.02)
        assert np.isclose(energies[2], E_21, rtol=0.02)

    def test_box_normalization(self):
        """Wavefunctions normalized: ∫∫|ψ|² dx dy = 1."""
        box = ParticleInBox2D(Lx=1e-9, Ly=1e-9)
        solver = SchrodingerSolver2D(
            box,
            x_range=(0, 1e-9),
            y_range=(0, 1e-9),
            num_points_x=80,
            num_points_y=80,
            num_states=3
        )

        _, wavefunctions = solver.solve()

        for i in range(wavefunctions.shape[2]):
            psi = wavefunctions[:, :, i]
            integral = np.sum(np.abs(psi)**2) * solver.dx * solver.dy
            assert np.isclose(integral, 1.0, rtol=1e-3)


class TestHarmonicOscillator2D:
    """Test 2D harmonic oscillator solver."""

    def test_qho_ground_state_energy(self):
        """Ground state E_0,0 = ℏω."""
        omega = 1e15  # rad/s
        qho = HarmonicOscillator2D(omega=omega)
        solver = SchrodingerSolver2D(
            qho,
            x_range=(-1e-9, 1e-9),
            y_range=(-1e-9, 1e-9),
            num_points_x=120,
            num_points_y=120,
            num_states=5
        )

        energies, _ = solver.solve()
        E_num = energies[0]
        E_ana = qho.energy_analytic(0, 0)

        # ~2% error for QHO (finite domain + grid)
        assert np.isclose(E_num, E_ana, rtol=0.03)

    def test_qho_energy_levels(self):
        """2D QHO energy levels: E_nx,ny = ℏω(nx + ny + 1)."""
        omega = 1e14
        qho = HarmonicOscillator2D(omega=omega)
        solver = SchrodingerSolver2D(
            qho,
            x_range=(-1.5e-9, 1.5e-9),
            y_range=(-1.5e-9, 1.5e-9),
            num_points_x=120,
            num_points_y=120,
            num_states=10
        )

        energies, _ = solver.solve()

        # Expected sequence: 1, 2, 2, 3, 3, 3, 4, 4, 4, 4 (degenerate sets)
        # States with nx+ny=0: 1 state (0,0)
        # States with nx+ny=1: 2 states (1,0), (0,1) -- degenerate
        # States with nx+ny=2: 3 states -- degenerate
        # etc.

        hw = hbar * omega

        # Ground state
        assert np.isclose(energies[0], hw, rtol=0.03)

        # First excited level (2-fold degenerate)
        assert np.isclose(energies[1], 2*hw, rtol=0.05)
        assert np.isclose(energies[2], 2*hw, rtol=0.05)

        # Second excited level (3-fold degenerate)
        assert np.isclose(energies[3], 3*hw, rtol=0.05)
        assert np.isclose(energies[4], 3*hw, rtol=0.05)
        assert np.isclose(energies[5], 3*hw, rtol=0.05)

    def test_qho_normalization(self):
        """2D QHO wavefunctions are normalized."""
        qho = HarmonicOscillator2D(omega=1e15)
        solver = SchrodingerSolver2D(
            qho,
            x_range=(-1e-9, 1e-9),
            y_range=(-1e-9, 1e-9),
            num_points_x=100,
            num_points_y=100,
            num_states=4
        )

        _, wavefunctions = solver.solve()

        for i in range(wavefunctions.shape[2]):
            psi = wavefunctions[:, :, i]
            integral = np.sum(np.abs(psi)**2) * solver.dx * solver.dy
            assert np.isclose(integral, 1.0, rtol=1e-3)


class TestProbabilityDensity2D:
    """Test 2D probability density properties."""

    def test_probability_density_positive(self):
        """ρ(x,y) = |ψ(x,y)|² always ≥ 0."""
        box = ParticleInBox2D(1e-9, 1e-9)
        solver = SchrodingerSolver2D(
            box,
            x_range=(0, 1e-9),
            y_range=(0, 1e-9),
            num_points_x=80,
            num_points_y=80,
            num_states=2
        )

        _, wavefunctions = solver.solve()

        for i in range(wavefunctions.shape[2]):
            rho = solver.probability_density(wavefunctions[:, :, i])
            assert np.all(rho >= -1e-14)

    def test_probability_density_integrates_to_one(self):
        """∫∫ρ(x,y) dx dy = 1."""
        box = ParticleInBox2D(1e-9, 1e-9)
        solver = SchrodingerSolver2D(
            box,
            x_range=(0, 1e-9),
            y_range=(0, 1e-9),
            num_points_x=80,
            num_points_y=80,
            num_states=2
        )

        _, wavefunctions = solver.solve()

        for i in range(wavefunctions.shape[2]):
            rho = solver.probability_density(wavefunctions[:, :, i])
            integral = np.sum(rho) * solver.dx * solver.dy
            assert np.isclose(integral, 1.0, rtol=1e-3)


class TestCustomPotential2D:
    """Test solver with user-defined 2D potentials."""

    def test_custom_potential_evaluation(self):
        """Solver works with custom 2D potentials."""
        def potential_func(x, y):
            X, Y = np.meshgrid(x, y)
            return 0.5 * (X**2 + Y**2)  # Simple harmonic-like

        custom = CustomPotential2D(potential_func, name_str="Custom 2D")
        solver = SchrodingerSolver2D(
            custom,
            x_range=(-1e-9, 1e-9),
            y_range=(-1e-9, 1e-9),
            num_points_x=100,
            num_points_y=100,
            num_states=3
        )

        energies, wavefunctions = solver.solve()

        # Should get valid solutions
        assert len(energies) >= 3
        assert wavefunctions.shape[:2] == (100, 100)
        assert wavefunctions.shape[2] >= 3


class TestSymmetry2D:
    """Test spatial symmetry properties of 2D states."""

    def test_ground_state_symmetric(self):
        """Ground state of 2D box is symmetric under (x↔y) swap."""
        box = ParticleInBox2D(1e-9, 1e-9)  # Square box
        solver = SchrodingerSolver2D(
            box,
            x_range=(0, 1e-9),
            y_range=(0, 1e-9),
            num_points_x=100,
            num_points_y=100,
            num_states=1
        )

        _, wavefunctions = solver.solve()
        psi = wavefunctions[:, :, 0]

        # Ground state should be symmetric
        # Check that psi(x, y) ≈ psi(y, x)
        psi_swapped = psi.T
        correlation = np.corrcoef(psi.flatten(), psi_swapped.flatten())[0, 1]
        assert correlation > 0.95

    def test_first_excited_degenerate_states(self):
        """(2,1) and (1,2) states are nearly degenerate for square box."""
        box = ParticleInBox2D(1e-9, 1e-9)
        solver = SchrodingerSolver2D(
            box,
            x_range=(0, 1e-9),
            y_range=(0, 1e-9),
            num_points_x=100,
            num_points_y=100,
            num_states=5
        )

        energies, _ = solver.solve()

        # For square box with E ∝ (nx² + ny²):
        # States 0: (1,1) E ∝ 2
        # States 1-2: (2,1) and (1,2) E ∝ 5 (degenerate)
        # States 3-4: (3,1), (1,3), (2,2) E ∝ 10, 10, 8

        E_11 = box.energy_analytic(1, 1)
        E_21 = box.energy_analytic(2, 1)  # Same as E_1,2

        # Ground state should match
        assert np.isclose(energies[0], E_11, rtol=0.02)

        # Next two states should be close to E_2,1 (nearly degenerate)
        assert np.isclose(energies[1], E_21, rtol=0.05)
        assert np.isclose(energies[2], E_21, rtol=0.05)


class TestAnalyticComparison:
    """Test retrieval of analytic energies."""

    def test_analytic_particle_in_box_2d(self):
        """Can retrieve analytic 2D box energies."""
        box = ParticleInBox2D(1e-9, 1e-9)
        solver = SchrodingerSolver2D(
            box,
            x_range=(0, 1e-9),
            y_range=(0, 1e-9),
            num_points_x=100,
            num_points_y=100
        )

        E_11 = solver.energy_analytic(1, 1)
        E_21 = solver.energy_analytic(2, 1)
        E_12 = solver.energy_analytic(1, 2)

        assert E_11 == box.energy_analytic(1, 1)
        assert E_21 == box.energy_analytic(2, 1)
        assert E_12 == box.energy_analytic(1, 2)

    def test_analytic_harmonic_oscillator_2d(self):
        """Can retrieve analytic 2D QHO energies."""
        qho = HarmonicOscillator2D(omega=1e15)
        solver = SchrodingerSolver2D(
            qho,
            x_range=(-1e-9, 1e-9),
            y_range=(-1e-9, 1e-9),
            num_points_x=100,
            num_points_y=100
        )

        E_00 = solver.energy_analytic(0, 0)
        E_10 = solver.energy_analytic(1, 0)
        E_01 = solver.energy_analytic(0, 1)

        assert E_00 == qho.energy_analytic(0, 0)
        assert E_10 == qho.energy_analytic(1, 0)
        assert E_01 == qho.energy_analytic(0, 1)

    def test_analytic_custom_potential_nan(self):
        """Analytic energy is NaN for unknown potentials."""
        def custom_func(x, y):
            X, Y = np.meshgrid(x, y)
            return X**2 * Y**2

        custom = CustomPotential2D(custom_func)
        solver = SchrodingerSolver2D(
            custom,
            x_range=(-1e-9, 1e-9),
            y_range=(-1e-9, 1e-9),
            num_points_x=100,
            num_points_y=100
        )

        E = solver.energy_analytic(1, 1)
        assert np.isnan(E)
