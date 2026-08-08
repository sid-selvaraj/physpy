"""
PhysPy — an open-source Python toolkit for computational physics.

Simulate, visualize, and compute physics concepts across mechanics,
electromagnetism, quantum mechanics, and more, built on top of NumPy
and SciPy.
"""

__version__ = "0.1.0"

from physpy.mechanics import Projectile
from physpy.electromagnetism import (
    PointCharge, ElectricDipole, ElectricField,
    Resistor, Capacitor, Inductor, VoltageSource, CurrentSource, Circuit
)

__all__ = [
    "Projectile",
    "PointCharge", "ElectricDipole", "ElectricField",
    "Resistor", "Capacitor", "Inductor", "VoltageSource", "CurrentSource", "Circuit",
    "__version__",
]
