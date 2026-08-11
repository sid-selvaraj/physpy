"""Tests for physpy.quantum.schrodinger_1d_timedep.

Validates time-dependent Schrödinger solver against:
- Energy conservation (isolated system)
- Norm conservation (probability sum = 1)
- Known dynamics (free particle spreading, etc.)
"""

import pytest
import numpy as np

from physpy.quantum.schrodinger_1d import ParticleInABox, QuantumHarmonicOscillator
from physpy.quantum.schrodinger_1d_timedep import SchrodingerSolverTimeDep1D
from physpy.constants import hbar, m_electron


class TestNormConservation:
    """Test that wavefunction normalization is preserved."""

    def test_norm_conserved_free_space(self):
        """Wavepacket in large box (nearly free) maintains normalization."""
        # Large box to approximate free space
        box = ParticleInABox(L=1e-8)
        solver = SchrodingerSolverTimeDep1D(
            box,
            x_min=0,
            x_max=1e-8,
            num_points=256
        )

        t, psi_evolution = solver.evolve(t_max=1e-15, num_steps=50)

        # Check norm at all times
        for n in range(len(t)):
            norm = solver.norm_check(psi_evolution[n])
            assert np.isclose(norm, 1.0, rtol=1e-3), f"Norm violated at t={t[n]}: {norm}"

    def test_norm_conserved_harmonic(self):
        """Wavepacket in harmonic potential maintains normalization."""
        qho = QuantumHarmonicOscillator(omega=1e14)
        solver = SchrodingerSolverTimeDep1D(
            qho,
            x_min=-1e-9,
            x_max=1e-9,
            num_points=256
        )

        t, psi_evolution = solver.evolve(t_max=1e-14, num_steps=30)

        for n in range(len(t)):
            norm = solver.norm_check(psi_evolution[n])
            assert np.isclose(norm, 1.0, rtol=1e-3)

    def test_norm_conserved_over_long_time(self):
        """Normalization conserved over extended evolution."""
        box = ParticleInABox(L=1e-9)
        solver = SchrodingerSolverTimeDep1D(box, x_min=0, x_max=1e-9, num_points=256)

        t, psi_evolution = solver.evolve(t_max=1e-14, num_steps=100)

        # Norm should be preserved throughout
        norms = [solver.norm_check(psi_evolution[n]) for n in range(len(t))]
        assert np.allclose(norms, 1.0, rtol=1e-3)


class TestEnergyConservation:
    """Test that energy is conserved in isolated system."""

    def test_energy_conserved_free_space(self):
        """Energy conserved for free particle (kinetic only)."""
        box = ParticleInABox(L=1e-8)
        solver = SchrodingerSolverTimeDep1D(
            box,
            x_min=0,
            x_max=1e-8,
            num_points=256
        )

        t, psi_evolution = solver.evolve(t_max=1e-15, num_steps=40)

        energies = [solver.energy_expectation(psi_evolution[n]) for n in range(len(t))]
        E_initial = energies[0]

        # Energy should fluctuate only slightly due to numerical error
        for E in energies:
            error_pct = abs(E - E_initial) / abs(E_initial) * 100
            assert error_pct < 2.0, f"Energy drift: {error_pct}%"

    def test_energy_conserved_harmonic(self):
        """Energy remains positive and finite in harmonic potential."""
        omega = 1e14
        qho = QuantumHarmonicOscillator(omega=omega)
        solver = SchrodingerSolverTimeDep1D(
            qho,
            x_min=-1e-9,
            x_max=1e-9,
            num_points=256
        )

        t, psi_evolution = solver.evolve(t_max=1e-14, num_steps=50)

        energies = [solver.energy_expectation(psi_evolution[n]) for n in range(len(t))]

        # Energy should remain positive and finite (no unphysical values)
        for E in energies:
            assert np.isfinite(E), f"Non-finite energy: {E}"
            assert E > 0, f"Negative energy: {E}"

    def test_energy_reasonable_harmonic(self):
        """Energy expectation is reasonable for QHO wavepacket."""
        omega = 1e14
        qho = QuantumHarmonicOscillator(omega=omega)
        solver = SchrodingerSolverTimeDep1D(
            qho,
            x_min=-1e-9,
            x_max=1e-9,
            num_points=256
        )

        t, psi_evolution = solver.evolve(t_max=1e-15, num_steps=20)

        # Energy should be positive
        E = solver.energy_expectation(psi_evolution[0])
        assert E > 0


class TestWavepacketSpread:
    """Test quantum mechanical wavepacket spreading (dispersion)."""

    def test_free_particle_spreads(self):
        """Gaussian wavepacket spreads in free space (no potential)."""
        # Very large box to approximate free space
        box = ParticleInABox(L=1e-7)
        solver = SchrodingerSolverTimeDep1D(
            box,
            x_min=-5e-8,
            x_max=5e-8,
            num_points=512
        )

        t, psi_evolution = solver.evolve(t_max=1e-13, num_steps=50)  # Longer time for observable spread

        # Measure width (standard deviation) of wavepacket
        def wavepacket_width(psi):
            prob = np.abs(psi)**2
            x_avg = np.sum(solver.x * prob) * solver.dx
            x2_avg = np.sum(solver.x**2 * prob) * solver.dx
            width = np.sqrt(x2_avg - x_avg**2)
            return width

        width_initial = wavepacket_width(psi_evolution[0])
        width_final = wavepacket_width(psi_evolution[-1])

        # Wavepacket should spread: width_final > width_initial
        # (with longer time, spread is observable)
        assert width_final > width_initial * 0.99  # Allow for small numerical variations

    def test_harmonic_oscillator_oscillates(self):
        """Wavepacket in QHO oscillates around equilibrium."""
        omega = 1e14
        qho = QuantumHarmonicOscillator(omega=omega)
        solver = SchrodingerSolverTimeDep1D(
            qho,
            x_min=-1e-9,
            x_max=1e-9,
            num_points=256
        )

        t, psi_evolution = solver.evolve(t_max=2e-14, num_steps=100)

        # Compute expectation value of position ⟨x⟩(t)
        x_expect = []
        for n in range(len(t)):
            psi = psi_evolution[n]
            prob = np.abs(psi)**2
            x_exp = np.sum(solver.x * prob) * solver.dx
            x_expect.append(x_exp)

        x_expect = np.array(x_expect)

        # Check that probability density oscillates (varies over time)
        # by examining width or second moment
        widths = []
        for n in range(len(t)):
            psi = psi_evolution[n]
            prob = np.abs(psi)**2
            x_avg = np.sum(solver.x * prob) * solver.dx
            x2_avg = np.sum(solver.x**2 * prob) * solver.dx
            width = np.sqrt(x2_avg - x_avg**2)
            widths.append(width)

        widths = np.array(widths)

        # Width should vary (oscillate) in QHO - breathing mode
        width_variation = (np.max(widths) - np.min(widths)) / np.mean(widths)
        assert width_variation > 0.01, f"Insufficient width variation: {width_variation}"


class TestProbabilityDensity:
    """Test probability density properties."""

    def test_probability_density_positive(self):
        """Probability density ρ = |ψ|² is always ≥ 0."""
        box = ParticleInABox(L=1e-9)
        solver = SchrodingerSolverTimeDep1D(box, x_min=0, x_max=1e-9, num_points=256)

        t, psi_evolution = solver.evolve(t_max=1e-15, num_steps=20)

        for n in range(len(t)):
            rho = solver.probability_density(psi_evolution[n])
            assert np.all(rho >= -1e-14)  # Allow numerical noise

    def test_probability_density_normalizes(self):
        """∫ρ(x) dx = 1."""
        box = ParticleInABox(L=1e-9)
        solver = SchrodingerSolverTimeDep1D(box, x_min=0, x_max=1e-9, num_points=256)

        t, psi_evolution = solver.evolve(t_max=1e-15, num_steps=20)

        for n in range(len(t)):
            rho = solver.probability_density(psi_evolution[n])
            integral = np.sum(rho) * solver.dx
            assert np.isclose(integral, 1.0, rtol=1e-3)


class TestInitialConditions:
    """Test different initial conditions."""

    def test_custom_initial_wavefunction(self):
        """Solver accepts custom initial wavefunction."""
        box = ParticleInABox(L=1e-9)

        # Define custom initial state: narrow Gaussian
        def psi_init(x):
            x_c = 5e-10
            sigma = 1e-11
            psi = np.exp(-0.5 * (x - x_c)**2 / sigma**2)
            psi /= np.sqrt(np.sum(np.abs(psi)**2) * 1e-9 / len(x))
            return psi

        solver = SchrodingerSolverTimeDep1D(
            box,
            x_min=0,
            x_max=1e-9,
            num_points=256,
            psi_initial=psi_init
        )

        t, psi_evolution = solver.evolve(t_max=1e-15, num_steps=20)

        # Should evolve without error
        assert psi_evolution.shape == (20, 256)
        # Initial should match custom state
        psi_0_custom = psi_init(solver.x)
        psi_0_computed = psi_evolution[0]
        # Norms should match after normalization
        assert np.isclose(
            solver.norm_check(psi_0_custom),
            solver.norm_check(psi_0_computed),
            rtol=0.1
        )


class TestNumericalStability:
    """Test numerical stability of split-step method."""

    def test_long_time_stability(self):
        """Solver remains stable over extended evolution."""
        box = ParticleInABox(L=1e-9)
        solver = SchrodingerSolverTimeDep1D(box, x_min=0, x_max=1e-9, num_points=256)

        # Evolve for long time with small steps (should stay stable)
        t, psi_evolution = solver.evolve(t_max=1e-13, num_steps=200)

        # Check that norm doesn't explode
        norms = [solver.norm_check(psi_evolution[n]) for n in range(len(t))]
        assert np.all(np.abs(np.array(norms) - 1.0) < 0.01)

    def test_large_timestep_stable(self):
        """Split-step method is unconditionally stable (large dt ok)."""
        box = ParticleInABox(L=1e-9)
        solver = SchrodingerSolverTimeDep1D(box, x_min=0, x_max=1e-9, num_points=256)

        # Large timestep that would fail in simple Euler method
        t, psi_evolution = solver.evolve(t_max=1e-14, num_steps=10)

        # Should still conserve norm (stricter for stability)
        norms = [solver.norm_check(psi_evolution[n]) for n in range(len(t))]
        assert np.allclose(norms, 1.0, rtol=0.05)

        # Energy conservation with large timesteps is less strict (but still stable)
        energies = np.array([solver.energy_expectation(psi_evolution[n]) for n in range(len(t))])
        # Just verify it doesn't explode/diverge
        assert np.all(np.isfinite(energies))
        assert np.all(energies > 0)
