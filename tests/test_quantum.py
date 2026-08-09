"""Tests for physpy.quantum.schrodinger_1d.

Validates 1D Schrödinger solver against known analytic solutions
for particle in a box and quantum harmonic oscillator.
"""

import pytest
import numpy as np

from physpy.quantum.schrodinger_1d import (
    SchrodingerSolver1D, ParticleInABox, QuantumHarmonicOscillator,
    FiniteSquareWell, CustomPotential
)
from physpy.constants import hbar, m_electron


class TestParticleInABox:
    """Test finite-difference solutions against analytic particle-in-a-box energies."""

    def test_box_energy_ground_state(self):
        """Numeric solution matches analytic E_1 for particle in a box."""
        L = 1e-9  # 1 nm box
        box = ParticleInABox(L=L)
        solver = SchrodingerSolver1D(
            box,
            x_min=0,
            x_max=L,
            num_points=1000,
            num_states=3
        )

        energies, _ = solver.solve()
        E1_numeric = energies[0]
        E1_analytic = box.energy_analytic(1)

        # Finite difference has ~O(dx²) error; 1% tolerance is reasonable
        assert np.isclose(E1_numeric, E1_analytic, rtol=0.01)

    def test_box_energy_excited_states(self):
        """Numeric energies satisfy E_n ∝ n² for particle in a box."""
        L = 1e-9
        box = ParticleInABox(L=L)
        solver = SchrodingerSolver1D(
            box,
            x_min=0,
            x_max=L,
            num_points=1000,
            num_states=5
        )

        energies, _ = solver.solve()

        # Check ratio E_2/E_1 ≈ 4, E_3/E_1 ≈ 9, etc.
        for n in range(1, 4):
            E_n_numeric = energies[n - 1]
            E_1_numeric = energies[0]
            expected_ratio = n**2
            actual_ratio = E_n_numeric / E_1_numeric
            assert np.isclose(actual_ratio, expected_ratio, rtol=0.02)

    def test_box_normalization(self):
        """Wavefunctions are properly normalized: ∫|ψ|² dx = 1."""
        L = 1e-9
        box = ParticleInABox(L=L)
        solver = SchrodingerSolver1D(box, x_min=0, x_max=L, num_points=500)

        _, wavefunctions = solver.solve()

        for i in range(wavefunctions.shape[1]):
            psi = wavefunctions[:, i]
            # Numerical integration: ∫|ψ|² dx ≈ Σ|ψ|² * dx
            integral = np.sum(np.abs(psi)**2) * solver.dx
            assert np.isclose(integral, 1.0, rtol=1e-3)

    def test_box_wavefunction_boundary(self):
        """Wavefunctions vanish at box boundaries (hard-wall BC)."""
        L = 1e-9
        box = ParticleInABox(L=L)
        solver = SchrodingerSolver1D(box, x_min=0, x_max=L, num_points=500)

        _, wavefunctions = solver.solve()

        # First and last points should be near zero (finite-diff boundary effect)
        for i in range(wavefunctions.shape[1]):
            psi = wavefunctions[:, i]
            # Amplitude at boundaries should be much smaller than interior
            interior_max = np.max(np.abs(psi[50:-50]))
            boundary_val = (np.abs(psi[0]) + np.abs(psi[-1])) / 2
            assert boundary_val < 0.3 * interior_max


class TestQuantumHarmonicOscillator:
    """Test QHO solver against analytic solutions."""

    def test_qho_ground_state_energy(self):
        """Numeric solution matches E_0 = 0.5*ℏ*ω for QHO."""
        omega = 1e15  # rad/s
        qho = QuantumHarmonicOscillator(omega=omega)
        solver = SchrodingerSolver1D(
            qho,
            x_min=-1e-9,
            x_max=1e-9,
            num_points=1000,
            num_states=3
        )

        energies, _ = solver.solve()
        E0_numeric = energies[0]
        E0_analytic = qho.energy_analytic(0)

        # QHO: ~1-2% error due to finite domain and grid
        assert np.isclose(E0_numeric, E0_analytic, rtol=0.02)

    def test_qho_energy_spacing(self):
        """QHO energy levels are evenly spaced: ΔE = ℏ*ω."""
        omega = 1e15
        qho = QuantumHarmonicOscillator(omega=omega)
        solver = SchrodingerSolver1D(
            qho,
            x_min=-1e-9,
            x_max=1e-9,
            num_points=1000,
            num_states=4
        )

        energies, _ = solver.solve()
        expected_spacing = hbar * omega

        # Check spacing between consecutive levels
        for i in range(len(energies) - 1):
            actual_spacing = energies[i + 1] - energies[i]
            assert np.isclose(actual_spacing, expected_spacing, rtol=0.05)

    def test_qho_normalization(self):
        """QHO wavefunctions are normalized."""
        omega = 1e15
        qho = QuantumHarmonicOscillator(omega=omega)
        solver = SchrodingerSolver1D(
            qho,
            x_min=-1e-9,
            x_max=1e-9,
            num_points=500,
            num_states=3
        )

        _, wavefunctions = solver.solve()

        for i in range(wavefunctions.shape[1]):
            psi = wavefunctions[:, i]
            integral = np.sum(np.abs(psi)**2) * solver.dx
            assert np.isclose(integral, 1.0, rtol=1e-3)


class TestFiniteSquareWell:
    """Test finite square well: wavefunctions tunnel outside."""

    def test_well_bound_states_exist(self):
        """Finite well supports bound states with E < V0."""
        L = 1e-9
        V0 = 100 * (hbar**2 * np.pi**2 / (2 * m_electron * L**2))  # Normalized depth
        well = FiniteSquareWell(L=L, V0=V0)

        solver = SchrodingerSolver1D(
            well,
            x_min=-L,
            x_max=L,
            num_points=800,
            num_states=3
        )

        energies, _ = solver.solve()

        # All bound states must have E < V0
        assert np.all(energies < V0)

    def test_well_wavefunction_penetration(self):
        """Wavefunctions penetrate outside finite square well."""
        L = 1e-9
        V0 = 50 * (hbar**2 * np.pi**2 / (2 * m_electron * L**2))
        well = FiniteSquareWell(L=L, V0=V0)

        solver = SchrodingerSolver1D(
            well,
            x_min=-L,
            x_max=L,
            num_points=800,
            num_states=1
        )

        _, wavefunctions = solver.solve()
        psi = wavefunctions[:, 0]

        # Find interior region |x| < L/2
        interior_mask = np.abs(solver.x) < L / 2
        exterior_mask = np.abs(solver.x) >= L / 2

        interior_max = np.max(np.abs(psi[interior_mask]))
        exterior_max = np.max(np.abs(psi[exterior_mask]))

        # Penetration: exterior amplitude > 0 but less than interior
        assert exterior_max > 0  # Tunneling occurs
        assert exterior_max < interior_max  # But weaker than interior


class TestCustomPotential:
    """Test solver with user-defined potentials."""

    def test_custom_potential_evaluation(self):
        """Solver works with custom potential functions."""
        def step_potential(x):
            """Step potential: increases at x=0."""
            return np.where(x < 0, 0, 1e-18)

        custom = CustomPotential(step_potential, name_str="Step Function")
        solver = SchrodingerSolver1D(
            custom,
            x_min=-1e-9,
            x_max=1e-9,
            num_points=500,
            num_states=2
        )

        energies, wavefunctions = solver.solve()

        # Should get valid solutions
        assert len(energies) >= 2
        assert wavefunctions.shape[0] == 500
        assert wavefunctions.shape[1] >= 2


class TestProbabilityDensity:
    """Test probability density calculations."""

    def test_probability_density_positive(self):
        """Probability density ρ = |ψ|² is always non-negative."""
        box = ParticleInABox(L=1e-9)
        solver = SchrodingerSolver1D(box, x_min=0, x_max=1e-9, num_points=500)

        _, wavefunctions = solver.solve()

        for i in range(wavefunctions.shape[1]):
            psi = wavefunctions[:, i]
            rho = solver.probability_density(psi)
            assert np.all(rho >= -1e-14)  # Allow for numerical errors

    def test_probability_density_integral(self):
        """∫ρ(x) dx = ∫|ψ(x)|² dx = 1."""
        box = ParticleInABox(L=1e-9)
        solver = SchrodingerSolver1D(box, x_min=0, x_max=1e-9, num_points=500)

        _, wavefunctions = solver.solve()

        for i in range(wavefunctions.shape[1]):
            psi = wavefunctions[:, i]
            rho = solver.probability_density(psi)
            integral = np.sum(rho) * solver.dx
            assert np.isclose(integral, 1.0, rtol=1e-3)


class TestEnergyAnalytic:
    """Test retrieval of analytic energy values."""

    def test_analytic_particle_in_box(self):
        """Can retrieve analytic energies for particle in box."""
        box = ParticleInABox(L=1e-9)
        solver = SchrodingerSolver1D(box, x_min=0, x_max=1e-9, num_points=500)

        for n in [1, 2, 3]:
            E_ana = solver.energy_analytic(n)
            E_expected = box.energy_analytic(n)
            assert E_ana == E_expected

    def test_analytic_harmonic_oscillator(self):
        """Can retrieve analytic energies for QHO."""
        qho = QuantumHarmonicOscillator(omega=1e15)
        solver = SchrodingerSolver1D(qho, x_min=-1e-9, x_max=1e-9, num_points=500)

        for n in [0, 1, 2]:
            E_ana = solver.energy_analytic(n)
            E_expected = qho.energy_analytic(n)
            assert E_ana == E_expected

    def test_analytic_custom_potential_nan(self):
        """Analytic energy is NaN for unknown potentials."""
        def custom_func(x):
            return x**4

        custom = CustomPotential(custom_func)
        solver = SchrodingerSolver1D(custom, x_min=-1e-9, x_max=1e-9, num_points=500)

        E_ana = solver.energy_analytic(1)
        assert np.isnan(E_ana)
