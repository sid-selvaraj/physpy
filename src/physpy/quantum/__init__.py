"""
physpy.quantum
==============
Quantum mechanics simulations and solvers.

Includes:
- **1D Schrödinger** — time-independent Schrödinger equation solver for 1D potentials
  - Particle in a box, harmonic oscillator, quantum wells
  - Eigenvalue problems using sparse matrix methods
  - Wavefunction visualization and probability densities
"""

from physpy.quantum.schrodinger_1d import (
    SchrodingerSolver1D, ParticleInABox, QuantumHarmonicOscillator,
    FiniteSquareWell, CustomPotential
)

__all__ = [
    "SchrodingerSolver1D",
    "ParticleInABox",
    "QuantumHarmonicOscillator",
    "FiniteSquareWell",
    "CustomPotential",
]
