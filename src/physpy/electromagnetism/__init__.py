"""
physpy.electromagnetism
=======================
Tools for simulating and visualizing electromagnetic phenomena.

Includes:
- **Statics** — point charges, dipoles, and electric field visualization
- **Circuits** — DC circuit solver using Kirchhoff's laws
"""

from physpy.electromagnetism.statics import PointCharge, ElectricDipole, ElectricField
from physpy.electromagnetism.circuits import (
    Resistor, Capacitor, Inductor, VoltageSource, CurrentSource, Circuit
)

__all__ = [
    "PointCharge", "ElectricDipole", "ElectricField",
    "Resistor", "Capacitor", "Inductor", "VoltageSource", "CurrentSource", "Circuit",
]
