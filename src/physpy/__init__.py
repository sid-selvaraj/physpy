"""
PhysPy — an open-source Python toolkit for computational physics.

Simulate, visualize, and compute physics concepts across mechanics,
electromagnetism, quantum mechanics, and more, built on top of NumPy
and SciPy. Every numerical result is validated against known analytic
solutions or physical laws.
"""

__version__ = "0.1.0"

from physpy.mechanics import Projectile
from physpy.mechanics.nbody import Body, NBodySolver
from physpy.electromagnetism import (
    PointCharge, ElectricDipole, ElectricField,
    Resistor, Capacitor, Inductor, VoltageSource, CurrentSource, Circuit
)
from physpy.electromagnetism.circuits_ac import ACCircuit, ACVoltageSource, ACCurrentSource
from physpy.electromagnetism.waves_em import FDTDSolver1D
from physpy.quantum import (
    SchrodingerSolver1D, ParticleInABox, QuantumHarmonicOscillator,
    FiniteSquareWell, CustomPotential,
    SchrodingerSolverTimeDep1D,
    SchrodingerSolver2D, ParticleInBox2D, HarmonicOscillator2D, CustomPotential2D
)

__all__ = [
    # Mechanics
    "Projectile",
    # Electromagnetism
    "PointCharge", "ElectricDipole", "ElectricField",
    "Resistor", "Capacitor", "Inductor", "VoltageSource", "CurrentSource", "Circuit",
    # Quantum
    "SchrodingerSolver1D", "ParticleInABox", "QuantumHarmonicOscillator",
    "FiniteSquareWell", "CustomPotential",
    # Version
    "__version__",
]
