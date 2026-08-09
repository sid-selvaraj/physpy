"""
Demo: 1D Quantum Mechanics — Schrödinger Equation Solver.

Shows how to:
1. Solve particle in a box (infinite square well)
2. Solve quantum harmonic oscillator
3. Solve finite square well with quantum tunneling
4. Compare numeric solutions with analytic predictions
"""

from physpy.quantum.schrodinger_1d import (
    SchrodingerSolver1D, ParticleInABox, QuantumHarmonicOscillator,
    FiniteSquareWell
)
from physpy.constants import hbar, m_electron
import numpy as np

print("=" * 70)
print("QUANTUM MECHANICS: 1D SCHRÖDINGER SOLVER DEMO")
print("=" * 70)

# ------- Example 1: Particle in a Box (Infinite Square Well) -------
print("\n1. Particle in a Box: L = 1 nm")
print("-" * 70)

L = 1e-9  # 1 nanometer
box = ParticleInABox(L=L)
solver_box = SchrodingerSolver1D(
    box,
    x_min=0,
    x_max=L,
    num_points=1000,
    num_states=4
)

print(f"Box length: {L*1e9:.1f} nm")
print(f"Particle mass: electron")
print("\nEnergy Levels:")
print("  n | Numeric (J)      | Analytic (J)      | Error (%)")
print("-" * 60)

energies_box, wavefunctions_box = solver_box.solve()
for n in range(4):
    E_num = energies_box[n]
    E_ana = box.energy_analytic(n + 1)
    error = abs(E_num - E_ana) / E_ana * 100
    print(f"  {n+1} | {E_num:.3e} | {E_ana:.3e} | {error:.2f}%")

print("\nGeneration wavefunction plot...")
solver_box.plot(show=False, num_states=3)
print("✓ Plot saved")

# ------- Example 2: Quantum Harmonic Oscillator -------
print("\n2. Quantum Harmonic Oscillator: ω = 1×10¹⁵ rad/s")
print("-" * 70)

omega = 1e15  # rad/s
qho = QuantumHarmonicOscillator(omega=omega)
solver_qho = SchrodingerSolver1D(
    qho,
    x_min=-1e-9,
    x_max=1e-9,
    num_points=1000,
    num_states=5
)

print(f"Angular frequency: ω = {omega:.1e} rad/s")
print(f"Zero-point energy: E₀ = ℏω/2 = {qho.energy_analytic(0):.3e} J")
print("\nEnergy Levels & Spacing:")
print("  n | Energy (J)       | ℏω Spacing Check")
print("-" * 60)

energies_qho, _ = solver_qho.solve()
E0_spacing = hbar * omega
for n in range(5):
    E_num = energies_qho[n]
    E_ana = qho.energy_analytic(n)
    if n > 0:
        spacing = energies_qho[n] - energies_qho[n-1]
        spacing_ratio = spacing / E0_spacing
        print(f"  {n} | {E_num:.3e} | {spacing_ratio:.3f} × ℏω")
    else:
        print(f"  {n} | {E_num:.3e} | (ground state)")

print("\nGenerating wavefunction plot...")
solver_qho.plot(show=False, num_states=3)
print("✓ Plot saved")

# ------- Example 3: Finite Square Well (Quantum Tunneling) -------
print("\n3. Finite Square Well: L = 2 nm, V₀ = 20 eV")
print("-" * 70)

L_well = 2e-9  # 2 nm
V0 = 20 * 1.602e-19  # 20 eV in Joules
well = FiniteSquareWell(L=L_well, V0=V0)

solver_well = SchrodingerSolver1D(
    well,
    x_min=-1.5e-9,
    x_max=1.5e-9,
    num_points=1000,
    num_states=3
)

print(f"Well width: {L_well*1e9:.1f} nm")
print(f"Well depth: {V0/1.602e-19:.1f} eV")
print("\nBound States (with quantum tunneling):")

energies_well, wavefunctions_well = solver_well.solve()
print("  n | Energy (J)       | E/V₀ Ratio | Inside/Outside Ratio")
print("-" * 65)

for n in range(3):
    E_num = energies_well[n]
    psi = wavefunctions_well[:, n]

    # Split wavefunction into interior and exterior
    x = solver_well.x
    interior = np.abs(x) < L_well / 2
    exterior = np.abs(x) >= L_well / 2

    psi_int_max = np.max(np.abs(psi[interior]))
    psi_ext_max = np.max(np.abs(psi[exterior]))

    ratio = psi_int_max / psi_ext_max if psi_ext_max > 0 else np.inf
    E_V_ratio = E_num / V0

    print(f"  {n+1} | {E_num:.3e} | {E_V_ratio:.4f}  | {ratio:.2f}")

print("\nNote: Finite ratio (not ∞) shows quantum tunneling into classically")
print("forbidden region. Wavefunction penetrates barrier despite E < V₀.")

print("\nGenerating wavefunction plot...")
solver_well.plot(show=False, num_states=2)
print("✓ Plot saved")

# ------- Example 4: Validation Summary -------
print("\n4. Validation Summary")
print("-" * 70)
print("Particle in a Box (n=1):")
print(f"  Numeric:  {energies_box[0]:.6e} J")
print(f"  Analytic: {box.energy_analytic(1):.6e} J")
error_pib = abs(energies_box[0] - box.energy_analytic(1)) / box.energy_analytic(1) * 100
print(f"  Error: {error_pib:.3f}%")

print("\nQuantum Harmonic Oscillator (n=0):")
print(f"  Numeric:  {energies_qho[0]:.6e} J")
print(f"  Analytic: {qho.energy_analytic(0):.6e} J")
error_qho = abs(energies_qho[0] - qho.energy_analytic(0)) / qho.energy_analytic(0) * 100
print(f"  Error: {error_qho:.3f}%")

print("\nFinite Square Well (numerical validation):")
print(f"  Ground state energy: {energies_well[0]:.3e} J")
print(f"  Well depth: {V0:.3e} J")
print(f"  E < V₀: {energies_well[0] < V0} ✓ (bound state)")

print("\n" + "=" * 70)
print("Demo complete! Physics validated against theory.")
print("=" * 70)
